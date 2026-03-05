import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# === CONFIG ===
RESULTS_DIR = "results"  # root directory containing baseline/, hydra/, etc.
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Collect all CSVs ===
all_files = []
for root, dirs, files in os.walk(RESULTS_DIR):
    for f in files:
        if f.endswith(".csv"):
            all_files.append(os.path.join(root, f))

print(f" Found {len(all_files)} result files.")

# === Step 2: Merge all into a single DataFrame ===
frames = []
for f in all_files:
    df = pd.read_csv(f)
    df["source_file"] = f
    # Try to detect agent and phase from folder path
    parts = Path(f).parts
    agent = [p for p in parts if p.lower() in ["hydra", "tdmpc2", "costar", "dreamerv3", "baseline"]]
    phase = [p for p in parts if p.lower() in ["early", "mid", "final"]]
    df["agent"] = agent[0] if agent else "unknown"
    df["phase"] = phase[0] if phase else "unknown"
    frames.append(df)

full_df = pd.concat(frames, ignore_index=True)
print(" Combined dataframe shape:", full_df.shape)

# Save combined data
full_df.to_csv(os.path.join(OUTPUT_DIR, "all_results_combined.csv"), index=False)

# === Step 3: Metrics (based on your real CSV structure) ===
metrics = [
    "reward",
    "latency_ms",
    "energy_efficiency",
    "task_scheduling_score",
    "scalability_score",
    "resource_allocation_score"
]
print("Metrics used:", metrics)

# === Step 4: Summary stats per agent & phase ===
summary = (
    full_df.groupby(["agent", "phase"])[metrics]
    .agg(["mean", "std", "count"])
    .reset_index()
)
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_by_agent_phase.csv"), index=False)
print("Saved summary statistics per agent & phase.")

# === Step 5: Compute 95% Confidence Intervals ===
ci_results = []
for (agent, phase), group in full_df.groupby(["agent", "phase"]):
    for metric in metrics:
        vals = group[metric].dropna()
        if len(vals) > 1:
            mean = vals.mean()
            sem = stats.sem(vals)
            ci = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=sem)
            ci_results.append([agent, phase, metric, mean, ci[0], ci[1]])
df_ci = pd.DataFrame(ci_results, columns=["agent", "phase", "metric", "mean", "ci_lower", "ci_upper"])
df_ci.to_csv(os.path.join(OUTPUT_DIR, "confidence_intervals.csv"), index=False)
print(" Saved 95% confidence intervals.")

# === Step 6: Pairwise significance tests (Final phase only) ===
final_df = full_df[full_df["phase"] == "final"]
pairwise_results = []
for metric in metrics:
    agents = final_df["agent"].unique()
    for i, a1 in enumerate(agents):
        for a2 in agents[i + 1:]:
            v1 = final_df[final_df["agent"] == a1][metric].dropna()
            v2 = final_df[final_df["agent"] == a2][metric].dropna()
            if len(v1) > 2 and len(v2) > 2:
                stat, p = stats.ttest_ind(v1, v2, equal_var=False)
                pairwise_results.append([metric, a1, a2, p])
df_pvalues = pd.DataFrame(pairwise_results, columns=["metric", "agent_1", "agent_2", "p_value"])
df_pvalues.to_csv(os.path.join(OUTPUT_DIR, "pairwise_significance_tests.csv"), index=False)
print("Saved pairwise significance tests for Final phase.")

# === Step 7: Per-metric global summary ===
global_summary = full_df.groupby("agent")[metrics].mean().reset_index()
global_summary.to_csv(os.path.join(OUTPUT_DIR, "global_mean_summary.csv"), index=False)
print("Saved global summary (mean per agent across all phases).")

print("\nAnalysis complete!")
print("Output files generated in:", OUTPUT_DIR)
print(" - all_results_combined.csv")
print(" - summary_by_agent_phase.csv")
print(" - confidence_intervals.csv")
print(" - pairwise_significance_tests.csv")
print(" - global_mean_summary.csv")
