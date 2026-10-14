# VNF Placement with Deep Reinforcement Learning for Internet of Vehicles

Code and processed data for the evaluation of **Admission-Aware Sequential VNF Placement with Deep Reinforcement Learning for Internet of Vehicles**, by Tuan-Minh Pham, Anh-Duc Bui and Serge Fdida.

The package evaluates trained DDQN and PPO policies, with and without feasibility masking, on Fat-Tree and BCube networks. It includes 20 scenarios and eight model checkpoints. Evaluation runs on a CPU; SUMO and retraining are not required.

## Installation

Use Python 3.11. Open a terminal in this directory and create an environment:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run the evaluation

```sh
python run_evaluation.py
```

This runs 80 evaluations: two topologies, four policies and request counts from 10 to 100 in increments of 10. Execution takes approximately ten minutes on the tested CPU; runtime depends on the machine.

For a smaller run:

```sh
python run_evaluation.py --topologies fat_tree_k6 --demands 10 --output results_small
```

Available policies are `ddqn_masked`, `ddqn_unmasked`, `ppo_masked` and `ppo_unmasked`. Use `--policies` to select them, or `--help` to list all options. Output files with the same names are overwritten when an output directory is reused.

## Results

The default output directory is `results/`:

- `metrics.csv`: cost, acceptance ratio, delay and other aggregate metrics.
- `metrics.json`: aggregate metrics and individual accepted-request-event delays.
- `figures/`: 16 plots for the comparisons in Figures 3 and 4, grouped by topology.

Average cost and delay are calculated over accepted request-events. The lowest-delay-m metric uses the smaller accepted-event count of the masked/unmasked pair, then averages each policy's own m lowest delays. Both policies in the pair must be evaluated to produce this metric. Plot formatting may differ from the manuscript.

## Experimental settings

The reward is `-(compute_cost + 0.15 * incremental_delay)` for a valid placement, with a penalty of 50 for rejection. Masked policies check the current placement's type, resource, connectivity and delay constraints. Optional look-ahead checking of the remaining chain is not enabled.

The supplied checkpoints were trained before the evaluation resource profile was applied; no retraining is performed here. Saved DDQN training settings use 150 episodes, replay capacity 30,000 and epsilon decay per episode. PPO uses 100 episodes. This package evaluates the saved models; it does not rerun training or generate the convergence experiment in Figure 2.

Dataset parameters and source attribution are described in [DATASET.md](DATASET.md).

## Files

```text
nfv_drl_iov/       Environment and policy implementations
data/             Processed scenarios for both topologies
checkpoints/      Trained DDQN and PPO models
third_party/      LuST license notice
run_evaluation.py Evaluation and plotting
requirements.txt Python dependencies
```

Please cite the paper when using this code and retain the dataset attribution and included license notice.
