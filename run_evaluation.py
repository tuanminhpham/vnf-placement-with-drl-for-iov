"""Evaluate DDQN and PPO policies for VNF placement in IoV networks."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from nfv_drl_iov.agent import DQNAgent
from nfv_drl_iov.env import DPVEnv, OBSERVATION_SCHEMA_VERSION
from nfv_drl_iov.masking import policy_action_mask
from nfv_drl_iov.models import load_scenario
from nfv_drl_iov.ppo_agent import PPOAgent

ROOT = Path(__file__).resolve().parent
TOPOLOGIES = ('fat_tree_k6', 'bcube_n4_k2')
POLICIES = ('ddqn_masked', 'ddqn_unmasked', 'ppo_masked', 'ppo_unmasked')


def evaluate(topology: str, policy: str, demand: int) -> dict:
    model_path = ROOT / 'checkpoints' / topology / f'{policy}.pt'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    scenario = load_scenario(ROOT / 'data' / topology / f'demands_{demand}.json')
    mask_mode = 'strict' if policy.endswith('_masked') else 'none'
    env = DPVEnv(scenario, **checkpoint['reward_config'], compute_action_mask=mask_mode == 'strict')
    if (checkpoint['observation_schema_version'] != OBSERVATION_SCHEMA_VERSION
            or checkpoint['placement_node_ids'] != env.placement_node_ids
            or checkpoint['action_size'] != env.action_space.n):
        raise ValueError('Checkpoint and scenario are incompatible')
    agent_class = DQNAgent if policy.startswith('ddqn') else PPOAgent
    agent = agent_class(env.observation_size, env.action_space.n, device='cpu')
    agent.load(model_path)
    observation, info = env.reset()
    total_reward, steps = 0.0, 0
    accepted_delays = []
    current_event = None
    event_delay_start = 0.0
    started = time.perf_counter()
    while True:
        event = (int(env.t), int(env.current_task()[0]))
        if event != current_event:
            current_event = event
            event_delay_start = float(env.total_delay)
        mask = policy_action_mask(env, mask_mode, info)
        if not mask.any():
            observation, reward, terminated, truncated, info = env.reject_current_request()
        else:
            if policy.startswith('ddqn'):
                action = agent.select_action(observation, mask, epsilon=0.0)
            else:
                action, _, _ = agent.select_action(observation, mask, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if info.get('request_boundary', False):
            if current_event in env.accepted_request_events:
                accepted_delays.append(float(env.total_delay) - event_delay_start)
            current_event = None
        if terminated or truncated:
            break
    report = env.placement_report()
    accepted = report['accepted_request_events']
    if len(accepted_delays) != accepted:
        raise ValueError('Accepted event count and delay records disagree')
    return dict(topology=topology, policy=policy, demand=demand,
                **{k: report[k] for k in ('active_request_events', 'accepted_request_events',
                   'rejected_request_events', 'acceptance_ratio_pct', 'total_compute_cost',
                   'served_total_delay')},
                avg_system_cost=report['total_compute_cost'] / accepted if accepted else None,
                avg_delay_ms=report['served_total_delay'] / accepted if accepted else None,
                total_reward=total_reward, steps=steps, accepted_delays_ms=accepted_delays,
                elapsed_seconds=time.perf_counter() - started)


def add_lowest_m(records: list[dict]) -> None:
    for topology in TOPOLOGIES:
        for demand in sorted({r['demand'] for r in records}):
            for algorithm in ('ddqn', 'ppo'):
                pair = [r for r in records if r['topology'] == topology and r['demand'] == demand
                        and r['policy'].startswith(algorithm)]
                if len(pair) != 2:
                    continue
                m = min(r['accepted_request_events'] for r in pair)
                for row in pair:
                    row['lowest_m'] = m
                    row['lowest_m_delay_ms'] = float(np.mean(sorted(row['accepted_delays_ms'])[:m])) if m else None


def plot_results(records: list[dict], destination: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = {'avg_system_cost': 'Average system cost',
               'acceptance_ratio_pct': 'Acceptance ratio (%)',
               'avg_delay_ms': 'Average end-to-end delay (ms)',
               'lowest_m_delay_ms': 'Lowest-delay m: average delay (ms)'}
    for topology in TOPOLOGIES:
        for algorithm in ('ddqn', 'ppo'):
            subset = [r for r in records if r['topology'] == topology and r['policy'].startswith(algorithm)]
            if not subset:
                continue
            for metric, label in metrics.items():
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                for suffix, prefix, style in (('masked', 'FAS-', '-D'), ('unmasked', '', '--^')):
                    rows = sorted((r for r in subset if r['policy'] == f'{algorithm}_{suffix}'
                                   and r.get(metric) is not None), key=lambda r: r['demand'])
                    if rows:
                        ax.plot([r['demand'] for r in rows], [r[metric] for r in rows], style,
                                label=prefix + algorithm.upper())
                if not ax.lines:
                    plt.close(fig)
                    continue
                ax.set(xlabel='Number of service requests', ylabel=label,
                       title='Fat-Tree (k=6)' if topology == 'fat_tree_k6' else 'BCube (4,2)')
                ax.grid(alpha=0.3)
                ax.legend()
                fig.tight_layout()
                folder = destination / 'figures' / topology
                folder.mkdir(parents=True, exist_ok=True)
                fig.savefig(folder / f'{algorithm}_{metric}.png', dpi=300)
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demands', nargs='+', type=int, choices=range(10, 101, 10), default=list(range(10, 101, 10)))
    parser.add_argument('--topologies', nargs='+', choices=TOPOLOGIES, default=TOPOLOGIES)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for topology in dict.fromkeys(args.topologies):
        for demand in dict.fromkeys(args.demands):
            for policy in dict.fromkeys(args.policies):
                row = evaluate(topology, policy, demand)
                records.append(row)
                print(f'{topology} | {demand:3d} | {policy:13s} | acceptance={row["acceptance_ratio_pct"]:.2f}% | {row["elapsed_seconds"]:.2f}s', flush=True)
    add_lowest_m(records)
    (args.output / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    fields = [k for k in records[0] if k != 'accepted_delays_ms']
    with (args.output / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in records)
    plot_results(records, args.output)
    print(f'Saved {len(records)} evaluations to {args.output}', flush=True)


if __name__ == '__main__':
    main()
