from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import uuid
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

# Transition: một mẫu kinh nghiệm: state, action, reward, next_state, done, next_mask
@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray

# ReplayBuffer: bộ nhớ kinh nghiệm
# - push(): thêm transition
# - sample(): lấy batch random để train

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=int(capacity))
        self.random = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = self.random.sample(self.buffer, int(batch_size))
        return (
            np.asarray([item.state for item in batch], dtype=np.float32),
            np.asarray([item.action for item in batch], dtype=np.int64),
            np.asarray([item.reward for item in batch], dtype=np.float32),
            np.asarray([item.next_state for item in batch], dtype=np.float32),
            np.asarray([item.done for item in batch], dtype=np.float32),
            np.asarray([item.next_mask for item in batch], dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)

# DQNAgent.__init__() : tạo policy network, target network, optimizer, replay buffer
class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        replay_capacity: int = 50000,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_net = QNetwork(observation_size, action_size).to(self.device)
        self.target_net = QNetwork(observation_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.replay = ReplayBuffer(replay_capacity, seed)

    '''
    # select_action(): chọn action theo epsilon-greedy có mask
        - random trong các action hợp lệ khi explore
        - chọn Q-value cao nhất trong action hợp lệ khi exploit
    '''
    def select_action(self, observation: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            raise ValueError("action mask has no valid actions")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid_actions.tolist()))

        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs).cpu().numpy()[0]
        q_values = q_values.copy()
        q_values[~mask] = -1e9
        return int(np.argmax(q_values))

    # remember(): lưu transition vào replay buffer
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.replay.push(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_mask=next_mask,
            )
        )

    #  learn(): train một batch
    '''
    - lấy Q hiện tại;
    - dùng Double DQN để chọn action kế tiếp
    - dùng target network để tính target Q
    - cập nhật bằng squared loss đúng theo công thức DDQN trong bài báo.
    '''
    def learn(self, batch_size: int) -> float | None:
        if len(self.replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.replay.sample(batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_policy_q = self.policy_net(next_states_t)
            next_policy_q = next_policy_q.masked_fill(~next_masks_t, -1e9)
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            has_next_action = next_masks_t.any(dim=1)
            next_target_q = torch.where(
                has_next_action,
                next_target_q,
                torch.zeros_like(next_target_q),
            )
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_target_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # update_target(): copy trọng số policy network sang target network
    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # save() / load(): lưu và đọc model .pt
    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "action_size": self.action_size,
            "gamma": self.gamma,
        }
        if metadata:
            checkpoint.update(metadata)
        temp_path = output_path.with_name(
            f"{output_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(
                checkpoint,
                temp_path,
                _use_new_zipfile_serialization=False,
            )
            try:
                if output_path.exists():
                    output_path.unlink()
                temp_path.replace(output_path)
                return output_path
            except (OSError, PermissionError):
                fallback_path = output_path.with_name(
                    f"{output_path.stem}_{uuid.uuid4().hex}{output_path.suffix}"
                )
                temp_path.replace(fallback_path)
                return fallback_path
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
