from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _sanitize_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 1:
        masks = masks.unsqueeze(0)
    if not masks.any(dim=1).all():
        raise ValueError("action mask has no valid actions")
    return masks


class ActorCriticPPO(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size))
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        detach_value_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        logits = self.policy(features)
        if masks is not None:
            masks = _sanitize_masks(masks)
            logits = logits.masked_fill(~masks, -1e9)
        value_features = features.detach() if detach_value_features else features
        return logits, self.value(value_features).squeeze(1)


@dataclass(frozen=True)
class PPOTransition:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    mask: np.ndarray


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.0,
        detach_value_features: bool = False,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)
        self.detach_value_features = bool(detach_value_features)
        self.learning_rate = float(learning_rate)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.network = ActorCriticPPO(observation_size, action_size).to(self.device)
        self.reset_optimizer()
        self.rollout: List[PPOTransition] = []
        self.bc_observations: torch.Tensor | None = None
        self.bc_actions: torch.Tensor | None = None
        self.bc_masks: torch.Tensor | None = None

    def reset_optimizer(self) -> None:
        """Drop optimizer momentum while preserving the current network weights."""
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

    def set_optimizer_learning_rate(self, learning_rate: float) -> None:
        """Temporarily change the active optimizer learning rate."""
        learning_rate = float(learning_rate)
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def select_action(
        self, observation: np.ndarray, mask: np.ndarray, deterministic: bool = False
    ) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(obs_t, mask_t)
            dist = Categorical(logits=logits)
            action_t = logits.argmax(dim=1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action_t)
        return int(action_t.item()), float(log_prob.item()), float(value.item())

    def predict_value(self, observation: np.ndarray, mask: np.ndarray) -> float:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(obs_t, mask_t)
        return float(value.item())

    def behavior_clone(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        batch_size: int,
        epochs: int,
    ) -> dict[str, float]:
        sample_count = self.set_behavior_clone_reference(
            observations,
            actions,
            masks,
        )
        assert self.bc_observations is not None
        assert self.bc_actions is not None
        assert self.bc_masks is not None
        observations_t = self.bc_observations
        actions_t = self.bc_actions
        masks_t = self.bc_masks

        batch_size = max(1, min(int(batch_size), sample_count))
        epochs = max(1, int(epochs))
        losses: list[float] = []
        accuracies: list[float] = []
        for _ in range(epochs):
            indices = np.arange(sample_count)
            self.rng.shuffle(indices)
            for start in range(0, sample_count, batch_size):
                batch_idx = torch.as_tensor(
                    indices[start : start + batch_size],
                    dtype=torch.int64,
                    device=self.device,
                )
                logits, _ = self.network(observations_t[batch_idx], masks_t[batch_idx])
                loss = F.cross_entropy(logits, actions_t[batch_idx])

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == actions_t[batch_idx]).float().mean()
                losses.append(float(loss.item()))
                accuracies.append(float(accuracy.item()))

        return {
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accuracies)),
            "samples": float(sample_count),
            "epochs": float(epochs),
        }

    def set_behavior_clone_reference(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
    ) -> int:
        observations_t = torch.as_tensor(
            np.asarray(observations, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions_t = torch.as_tensor(
            np.asarray(actions, dtype=np.int64),
            dtype=torch.int64,
            device=self.device,
        )
        masks_t = torch.as_tensor(
            np.asarray(masks, dtype=bool),
            dtype=torch.bool,
            device=self.device,
        )
        sample_count = int(actions_t.shape[0])
        if sample_count == 0:
            raise ValueError("behavior cloning requires at least one demonstration")
        if torch.any(actions_t < 0) or torch.any(actions_t >= masks_t.shape[1]):
            raise ValueError("behavior cloning action is outside the action space")
        target_is_available = masks_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        if not bool(torch.all(target_is_available)):
            raise ValueError("behavior cloning target must be enabled by its action mask")

        self.bc_observations = observations_t
        self.bc_actions = actions_t
        self.bc_masks = masks_t
        return sample_count

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        mask: np.ndarray,
    ) -> None:
        self.rollout.append(
            PPOTransition(
                state=state,
                action=int(action),
                reward=float(reward),
                done=bool(done),
                log_prob=float(log_prob),
                value=float(value),
                mask=np.asarray(mask, dtype=bool).copy(),
            )
        )

    def learn(
        self,
        batch_size: int,
        update_epochs: int,
        last_value: float = 0.0,
        bc_aux_coef: float = 0.0,
        bc_batch_size: int = 64,
    ) -> dict[str, float] | None:
        if not self.rollout:
            return None
        bc_aux_coef = float(bc_aux_coef)
        if bc_aux_coef < 0.0:
            raise ValueError("behavior-cloning auxiliary coefficient must be non-negative")
        if bc_aux_coef > 0.0 and (
            self.bc_observations is None
            or self.bc_actions is None
            or self.bc_masks is None
        ):
            raise ValueError(
                "behavior-cloning auxiliary loss requires a reference dataset"
            )

        states_t = torch.as_tensor(
            np.asarray([item.state for item in self.rollout], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions_t = torch.as_tensor(
            [item.action for item in self.rollout], dtype=torch.int64, device=self.device
        )
        rewards_t = torch.as_tensor(
            [item.reward for item in self.rollout], dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(
            [item.done for item in self.rollout], dtype=torch.float32, device=self.device
        )
        old_log_probs_t = torch.as_tensor(
            [item.log_prob for item in self.rollout], dtype=torch.float32, device=self.device
        )
        values_t = torch.as_tensor(
            [item.value for item in self.rollout], dtype=torch.float32, device=self.device
        )
        masks_t = torch.as_tensor(
            np.asarray([item.mask for item in self.rollout], dtype=bool),
            dtype=torch.bool,
            device=self.device,
        )

        advantages_t, returns_t = self._compute_advantages(
            rewards_t, dones_t, values_t, float(last_value)
        )
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std(unbiased=False) + 1e-8
            )

        sample_count = len(self.rollout)
        batch_size = max(1, min(int(batch_size), sample_count))
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "bc_aux_loss": [],
        }

        update_epochs = max(1, int(update_epochs))
        optimizer_steps = 0
        kl_early_stop = False
        for _ in range(update_epochs):
            indices = np.arange(sample_count)
            self.rng.shuffle(indices)
            for start in range(0, sample_count, batch_size):
                batch_idx = torch.as_tensor(
                    indices[start : start + batch_size], dtype=torch.int64, device=self.device
                )
                logits, value_pred = self.network(
                    states_t[batch_idx],
                    masks_t[batch_idx],
                    detach_value_features=self.detach_value_features,
                )
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                unclipped = ratio * advantages_t[batch_idx]
                clipped = (
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    * advantages_t[batch_idx]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value_pred, returns_t[batch_idx])
                bc_aux_loss = torch.zeros((), device=self.device)
                if bc_aux_coef > 0.0:
                    assert self.bc_observations is not None
                    assert self.bc_actions is not None
                    assert self.bc_masks is not None
                    bc_sample_count = int(self.bc_actions.shape[0])
                    bc_batch_size = max(
                        1,
                        min(int(bc_batch_size), bc_sample_count),
                    )
                    bc_indices = torch.as_tensor(
                        self.rng.choice(
                            bc_sample_count,
                            size=bc_batch_size,
                            replace=False,
                        ),
                        dtype=torch.int64,
                        device=self.device,
                    )
                    bc_logits, _ = self.network(
                        self.bc_observations[bc_indices],
                        self.bc_masks[bc_indices],
                    )
                    bc_aux_loss = F.cross_entropy(
                        bc_logits,
                        self.bc_actions[bc_indices],
                    )
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + bc_aux_coef * bc_aux_loss
                )

                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs_t[batch_idx]
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()
                    )
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approx_kl"].append(float(approx_kl.item()))
                metrics["clip_fraction"].append(float(clip_fraction.item()))
                metrics["bc_aux_loss"].append(float(bc_aux_loss.item()))

                # Check before applying another optimizer step. This is especially
                # important after behavior cloning, where one PPO epoch can contain
                # many minibatches and quickly destroy the pretrained policy.
                if self.target_kl > 0.0 and float(approx_kl.item()) > self.target_kl:
                    kl_early_stop = True
                    break

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                optimizer_steps += 1
            if kl_early_stop:
                break

        self.rollout.clear()
        result = {key: float(np.mean(values)) for key, values in metrics.items()}
        result["optimizer_steps"] = float(optimizer_steps)
        result["kl_early_stop"] = float(kl_early_stop)
        result["bc_aux_coef"] = bc_aux_coef
        return result

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        last_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        next_value = torch.tensor(last_value, dtype=torch.float32, device=self.device)

        for step in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_value * nonterminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[step] = gae
            next_value = values[step]

        return advantages, advantages + values

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "algorithm": "masked_actor_critic_ppo",
            "network_state_dict": self.network.state_dict(),
            "action_size": self.action_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "detach_value_features": self.detach_value_features,
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
            except PermissionError:
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
        self.network.load_state_dict(checkpoint["network_state_dict"])
