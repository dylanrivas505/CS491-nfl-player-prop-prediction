# RL Player Props (CS491 Final Project)
Predict NFL player props with PPO over multiple seasons (2018–2023) using position-aware masking/weighting and feature variants (with/without rolling history). Sweeps cover learning rate, rollout length, clipping, network size, and feature sets; plots show mean ± std reward across seeds for the best configs and grouped by hyperparameter.

## Repo Layout
- `sweep_experiments.py`: runs exhaustive sweeps (162 configs × 3 seeds) and logs per-episode rewards to `sweep_episode_logs/`, per-run CSV to `sweep_runs_top.csv`, and aggregated stats to `sweep_results_top.csv`.
- `plot_sweep_curves.py`: aggregates per-episode logs and renders mean ± std reward curves (overall top-k or grouped by lr/clip/rollout/hidden/features) into `outplots/`.
- `nfl_ppo.py`: PPO implementation (actor-critic MLP, Gaussian policy, masked reward).
- `envs/nfl_env.py`, `load_stats.py`: environment and data loading with position-aware masking/normalization.

## Data & Setup
- Data: pulled via `nflreadpy` (2018–2023 seasons), normalized targets; position-aware masking so QBs aren’t graded on receiving, etc.
- Episodes: capped at 512 steps; each run collects ~95–100 episodes for 50k timesteps.
- Device: runs on CPU/GPU; MPS supported on Apple Silicon.

## How to Run
1) Install deps (ensure `matplotlib`, `pandas`, `numpy`, `torch`, `nflreadpy`):  
   `pip install -r requirements.txt` (or install manually if no requirements file).
2) Run sweeps (writes logs/CSVs):  
   `python sweep_experiments.py`
3) Plot reward curves (examples):  
   - Overall top 5: `python plot_sweep_curves.py --log-dir sweep_episode_logs --out-dir outplots --top-k 5 --smooth 5`  
   - By learning rate: `python plot_sweep_curves.py --group-by lr --top-per-group 1 --smooth 5`  
   - By clip: `python plot_sweep_curves.py --group-by clip --top-per-group 1 --smooth 5`  
   - By rollout: `python plot_sweep_curves.py --group-by rollout --top-per-group 1 --smooth 5`  
   - By hidden size: `python plot_sweep_curves.py --group-by hidden --top-per-group 1 --smooth 5`  
   - By feature variant: `python plot_sweep_curves.py --group-by fv --top-per-group 1 --smooth 5`

## Notable Findings (from current sweeps)
- Best configs use clip=0.25, rollout=1024, lr ∈ {5e-4, 3e-4}, hidden (256,256)/(512,512); rewards ≈204–207 with low std (~2–4).
- Feature variants are close: no_history slightly higher reward on average; full sometimes lower val MSE. Differences are small.
- Learning rate: 5e-4 > 3e-4 > 1e-4 for reward; 5e-4 also trains faster on average.
- Rollout: 1024 is the sweet spot; 2048 smoother but slightly lower reward; 512 slightly lower reward with higher variance.
- Clipping: 0.25 > 0.2 >> 0.1; looser clipping helped without runtime cost.
- Capacity: (256,256)/(512,512) outperform (128,128) modestly.

## Repro Notes
- Exhaustive configs: 162 (2 feature variants × 3 lrs × 3 rollouts × 3 hidden sizes × 3 clips × 1 temp × 1 max_ep), 3 seeds each (486 runs).
- Logs: per-episode CSVs in `sweep_episode_logs/`; per-run stats in `sweep_runs_top.csv`; aggregated in `sweep_results_top.csv`.
