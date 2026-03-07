# Reproducible DRL Orchestration Benchmark (Real Framework)

This repository is a **real, runnable orchestration benchmarking framework** designed to match the paper's *framework architecture*:
- Unified `observe()` and `act(action)` interfaces
- Trace-driven workload generation (Poisson arrivals + geometric bursts) for Early/Mid/Final phases
- Real Docker-based execution (no purely synthetic "environment-only" simulation)
- Per-episode CSV logging with timestamps, agent, phase, mode, seed, and all metrics
- Statistical analysis: Shapiro–Wilk, Welch t-test / Wilcoxon, Holm correction
- Figure generation (PDF) from CSV logs

## Important note about DRL agents (Hydra / TD-MPC2 / CoSTAR / DreamerV3)
The paper evaluates four specific algorithms. Those algorithms require their **official implementations** (separate codebases).
This repo includes **clean adapters** (see `agents/`) so you can plug in the official implementations and reproduce the study.
By default, adapters fall back to a simple built-in policy if an external agent implementation is not available.

That means:
- The **framework** is real and reproducible.
- To reproduce the paper **exactly**, you must connect the official agent code in `agents/external/` as described below.

## Quick start

### 1) Requirements
- Python 3.10+
- Docker Desktop running (Windows 11 supported)

Install Python deps:
```bash
pip install -r requirements.txt
```

### 2) Start workload containers
This framework controls a set of workload containers it starts/stops dynamically.
Build the workload image:
```bash
docker build -t wasm_workload:latest docker/workload
```

### 3) Run an experiment (reduced demo)
```bash
python runner.py --phase early --train-eps 50 --test-eps 50 --seeds 0 1
```

Full paper episode counts:
- Early: train 2000, test 4000
- Mid:   train 4000, test 6000
- Final: train 6000, test 10000

### 4) Analyze + chart
```bash
python analyze_results.py
python charts.py
```

## Plugging in official agent implementations
Place each official repo under:
- `agents/external/hydra/`
- `agents/external/tdmpc2/`
- `agents/external/costar/`
- `agents/external/dreamerv3/`

Then update `agents/external_config.yaml` with the command (or import path) used to query the policy:
- required interface: `predict(observation) -> action_index`


## Outputs
- `results/<agent>/<phase>/<mode>/seed_<k>.csv` (episode-level logs)
- `analysis/` (summary tables + statistical tests)
- `charts/` (PDF figures)

## Reproducibility
- All randomness is seeded per run and per agent/phase.

## Author
- Enes Bajrami
- PhD Candidate
- Ss. Cyril and Methodius University in Skopje, Faculty of Computer Science and Engineering, Skopje, North Macedonia
- enes.bajrami@students.finki.ukim.mk
