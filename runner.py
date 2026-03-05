import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import yaml

from env import OrchestrationEnv, RewardParams
from workload_engine import PhaseParams, generate_trace, load_trace
from agents.registry import make_agent

RESULTS_DIR = Path("results")

# Paper Table 7: Episode counts per experimental phase (baseline uses testing count).
PAPER_PHASE_EPISODES = {
    "early": {"train_eps": 2000, "test_eps": 4000, "baseline_eps": 4000},
    "mid":   {"train_eps": 4000, "test_eps": 6000, "baseline_eps": 6000},
    "final": {"train_eps": 6000, "test_eps": 10000, "baseline_eps": 10000},
}

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_csv(fp: Path, rows):
    ensure_dirs(fp.parent)
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def run_one(agent_name: str, phase: str, mode: str, seed: int, episodes: int, reward_set: str, trace_fp: Path):
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    rp = cfg["reward_sets"][reward_set]
    env = OrchestrationEnv(
        phase=phase,
        agent_name=agent_name,
        mode=mode,
        reward_params=RewardParams(**rp),
    )
    agent = make_agent(agent_name)
    agent.reset(seed)

    trace = load_trace(str(trace_fp))
    rows = []
    for ep in range(1, episodes + 1):
        obs = env.reset(seed=seed)
        workload = trace[ep - 1]
        action = agent.predict(obs)
        _, reward, done, info = env.step(action, workload)
        rows.append({"episode": ep, "reward": reward, **info})
        if ep % 200 == 0:
            print(f"  {agent_name} {phase} {mode} seed={seed} ep {ep}/{episodes}")
    return rows

def _get_phase_eps(cfg: dict, phase: str, args) -> tuple[int, int, int]:
    """
    Priority:
      1) CLI overrides (args.train_eps/args.test_eps)
      2) config.yaml if available
      3) paper defaults (Table 7)
    Returns: (train_eps, test_eps, baseline_eps)
    """
    paper = PAPER_PHASE_EPISODES[phase]

    cfg_phase = (cfg.get("phases", {}) or {}).get(phase, {}) or {}
    cfg_train = cfg_phase.get("train_eps", None)
    cfg_test  = cfg_phase.get("test_eps", None)

    train_eps = args.train_eps if args.train_eps is not None else (cfg_train if cfg_train is not None else paper["train_eps"])
    test_eps  = args.test_eps  if args.test_eps  is not None else (cfg_test  if cfg_test  is not None else paper["test_eps"])

    # Baseline runs for the testing horizon in the paper.
    baseline_eps = paper["baseline_eps"]
    # If user overrides test_eps, it is usually expected baseline aligns with that horizon.
    if args.test_eps is not None:
        baseline_eps = args.test_eps
    elif cfg_test is not None:
        baseline_eps = cfg_test

    return int(train_eps), int(test_eps), int(baseline_eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["early", "mid", "final"], default=None)
    ap.add_argument("--reward-set", choices=["C0", "S1", "S2", "S3"], default="C0")
    ap.add_argument("--agents", nargs="+", default=["hydra", "tdmpc2", "costar", "dreamerv3", "baseline"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-eps", type=int, default=None)
    ap.add_argument("--test-eps", type=int, default=None)
    ap.add_argument("--trace-duration", type=int, default=10)
    ap.add_argument("--trace-dir", default="traces")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    phases = [args.phase] if args.phase else ["early", "mid", "final"]

    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for phase in phases:
        pc = cfg["phases"][phase]
        train_eps, test_eps, baseline_eps = _get_phase_eps(cfg, phase, args)

        # Pre-generate traces per seed (replayed for all agents)
        trace_dir = Path(args.trace_dir)
        for seed in args.seeds:
            fp = trace_dir / f"{phase}_seed{seed}_trace.json"
            if not fp.exists() or fp.stat().st_size == 0:
                params = PhaseParams(
                    lam=pc["lambda"],
                    mean_burst=pc["mean_burst"],
                    duration_s=args.trace_duration,
                    mix=pc["mix"],
                )
                # Ensure trace covers the longest horizon needed this phase
                generate_trace(seed, phase, params, trace_dir, episodes=max(train_eps, test_eps, baseline_eps))

        for agent in args.agents:
            for seed in args.seeds:
                trace_fp = Path(args.trace_dir) / f"{phase}_seed{seed}_trace.json"

                if agent != "baseline":
                    print(f"TRAIN {agent} phase={phase} seed={seed} eps={train_eps}")
                    train_rows = run_one(agent, phase, "train", seed, train_eps, args.reward_set, trace_fp)
                    save_csv(RESULTS_DIR / agent / phase / "train" / f"seed_{seed}.csv", train_rows)

                    print(f"TEST  {agent} phase={phase} seed={seed} eps={test_eps}")
                    test_rows = run_one(agent, phase, "test", seed, test_eps, args.reward_set, trace_fp)
                    save_csv(RESULTS_DIR / agent / phase / "test" / f"seed_{seed}.csv", test_rows)
                else:
                    # Baseline: only test horizon (paper uses same count as testing)
                    print(f"TEST  {agent} phase={phase} seed={seed} eps={baseline_eps}")
                    base_rows = run_one(agent, phase, "test", seed, baseline_eps, args.reward_set, trace_fp)
                    save_csv(RESULTS_DIR / agent / phase / "test" / f"seed_{seed}.csv", base_rows)

    print("Done.")

if __name__ == "__main__":
    main()