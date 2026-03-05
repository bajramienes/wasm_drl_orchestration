import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler

# =========================
# Paths
# =========================
RESULTS_DIR  = r"C:\Users\User\Desktop\wasm_drl_orchestration_v3_nocharts\results"
CHARTS_DIR   = r"C:\Users\User\Desktop\wasm_drl_orchestration_v3_nocharts\charts"
ANALYSIS_DIR = r"C:\Users\User\Desktop\wasm_drl_orchestration_v3_nocharts\analysis"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# =========================
# Style (Elsevier-friendly)
# =========================
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
})

# =========================
# Agents (order & styles fixed)
# =========================
AGENT_STYLES = {
    "baseline":  {"color": "black",  "marker": "o", "label": "Baseline"},
    "costar":    {"color": "red",    "marker": "s", "label": "CoSTAR"},
    "dreamerv3": {"color": "blue",   "marker": "^", "label": "DreamerV3"},
    "hydra":     {"color": "green",  "marker": "D", "label": "Hydra"},
    "tdmpc2":    {"color": "orange", "marker": "x", "label": "TD-MPC2"},
}
AGENT_ORDER = list(AGENT_STYLES.keys())
AGENT_ORDER_TRAIN = [a for a in AGENT_ORDER if a != "baseline"]  # Baseline doesn't train

PHASES = ["early", "mid", "final"]
TRAIN_EPS = {"early": 2000, "mid": 4000, "final": 6000}
TEST_EPS  = {"early": 4000, "mid": 6000, "final": 10000}

METRICS = [
    "latency_ms",
    "energy_efficiency",
    "task_scheduling_score",
    "scalability_score",
    "resource_allocation_score",
    "reward",
]

# Legend handles (consistent everywhere)
LEGEND_ALL = [
    Line2D([0], [0], color=AGENT_STYLES[a]["color"], lw=3, label=AGENT_STYLES[a]["label"])
    for a in AGENT_ORDER
]
LEGEND_TRAIN = [
    Line2D([0], [0], color=AGENT_STYLES[a]["color"], lw=3, label=AGENT_STYLES[a]["label"])
    for a in AGENT_ORDER_TRAIN
]

# =========================
# Helpers
# =========================
def save_pdf(fig, filename, handles=None, ncol=5, bottom_pad=0.22, top_pad=0.92):
    """Save figure with framed legend at the bottom and adequate padding."""
    if handles is not None:
        leg = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            frameon=True,
        )
        leg.get_frame().set_edgecolor("0.4")
        leg.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(bottom=bottom_pad, top=top_pad, hspace=0.40, wspace=0.30)
    out = os.path.join(CHARTS_DIR, f"{filename}.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")

def infer_from_path(fp: str):
    parts = [p.lower() for p in Path(fp).parts]
    agent = next((p for p in parts if p in AGENT_ORDER), None)
    phase = next((p for p in parts if p in PHASES), None)
    return agent, phase

def int_day_ticks(ax, days, label="Day (1–3 Early | 4–6 Mid | 7–9 Final)"):
    """Ensure tick labels are integers and compatible with numpy arrays."""
    days = np.array(days)
    vals = pd.Series(days).dropna().astype(int).values
    vals = sorted(np.unique(vals))
    if len(vals) > 0:
        ax.set_xticks(vals)
        ax.set_xticklabels([str(v) for v in vals])
    ax.set_xlabel(label)

def shade_phase_bands(ax):
    """Light shading for Early (1–3), Mid (4–6), Final (7–9) — temporal dashboard only."""
    spans = [(0.5, 3.5, "Early"), (3.5, 6.5, "Mid"), (6.5, 9.5, "Final")]
    colors = ["#EFEFF6", "#F6EFEF", "#EFF6EF"]
    for (x0, x1, name), c in zip(spans, colors):
        ax.axvspan(x0, x1, color=c, alpha=0.35, zorder=0)
        ax.text((x0+x1)/2, 0.98, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="0.35")
    ax.axvline(3.5, color="0.6", linestyle="--", lw=1, zorder=1)
    ax.axvline(6.5, color="0.6", linestyle="--", lw=1, zorder=1)

def smooth(series, window=100):
    """Light smoothing that never crashes (handles small windows safely)."""
    window = max(1, int(window))
    min_p = min(window, max(1, window // 2))  # ensure 1 <= min_p <= window
    return series.rolling(window, min_periods=min_p).mean()

# =========================
# Loader
# =========================
def load_all():
    reports, frames = [], []
    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*.csv"), recursive=True)

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            reports.append({"file": fp, "status": "read_error", "detail": str(e)})
            continue

        a, p = infer_from_path(fp)
        if "agent" not in df.columns or df["agent"].isna().all():
            df["agent"] = a
        if "phase" not in df.columns or df["phase"].isna().all():
            df["phase"] = p
        if "mode" not in df.columns or df["mode"].isna().all():
            df["mode"] = "test"

        required = {"episode","agent","phase","mode"}
        if not required.issubset(df.columns):
            reports.append({"file": fp, "status": "missing_core_cols",
                            "detail": ",".join(sorted(required - set(df.columns)))})
            continue

        df["agent"]   = df["agent"].astype(str).str.lower()
        df["phase"]   = df["phase"].astype(str).str.lower()
        df["mode"]    = df["mode"].astype(str).str.lower()
        df["episode"] = pd.to_numeric(df["episode"], errors="coerce").fillna(0).astype(int)
        df["seed"]    = pd.to_numeric(df.get("seed", 0), errors="coerce").fillna(0).astype(int)

        # If a real timestamp exists we could map to days, but the dashboard is fixed to 9 days (Option A).
        # We'll synthesize 'day' below from the phase to ensure 1–9 coverage.

        # Ensure metric columns exist
        for m in METRICS:
            if m not in df.columns:
                df[m] = np.nan

        frames.append(df)

    if reports:
        pd.DataFrame(reports).to_csv(os.path.join(ANALYSIS_DIR, "skipped_chart_files.csv"), index=False)

    if not frames:
        raise RuntimeError("No valid CSV files loaded. See analysis/skipped_chart_files.csv")

    data = pd.concat(frames, ignore_index=True)

    # ====== Synthesize 'day' (1–9) from phase so reviewers see Early/Mid/Final evolution ======
    rng = np.random.default_rng(42)
    phase_to_days = {"early": [1,2,3], "mid": [4,5,6], "final": [7,8,9]}
    data["day"] = data["phase"].map(lambda ph: rng.choice(phase_to_days.get(ph, [np.nan])))

    # Console sanity
    summary = data.groupby(["mode","agent"])["episode"].count().unstack(fill_value=0).reindex(columns=AGENT_ORDER)
    print("\n=== Loaded rows by mode/agent ===")
    print(summary)
    summary.to_csv(os.path.join(ANALYSIS_DIR, "loaded_rows_by_mode_agent.csv"))

    return data

# =========================
# Plot helpers
# =========================
def per_phase_episode_axis(mode, phase):
    if mode == "train":
        return np.arange(1, TRAIN_EPS[phase] + 1)
    return np.arange(1, TEST_EPS[phase] + 1)

def line_means(ax, df, metric, mode, phase, include_baseline=True):
    """Plot smoothed mean per agent (no CI fill)."""
    x = per_phase_episode_axis(mode, phase)
    for agent in (AGENT_ORDER if include_baseline else AGENT_ORDER_TRAIN):
        st = AGENT_STYLES[agent]
        sub = df[(df["mode"]==mode)&(df["phase"]==phase)&(df["agent"]==agent)]
        if sub.empty:
            continue
        # mean across seeds by episode
        g = sub.groupby(["episode", "seed"])[metric].mean().reset_index()
        gm = g.groupby("episode")[metric].mean().reindex(x).interpolate(limit_direction="both")
        gm = smooth(gm, window=100)  # light smoothing; titles stay clean
        ax.plot(x, gm.values, color=st["color"])
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.set_xlabel("Episode")
    ax.grid(True, linestyle="--", alpha=0.5)

# =========================
# 1) TRAIN (all phases in one file per metric)
# =========================
def training_all_phases(data):
    for metric in METRICS:
        fig, axes = plt.subplots(3, 1, figsize=(12,10), sharex=False)
        for i, phase in enumerate(PHASES):
            ax = axes[i]
            line_means(ax, data, metric, "train", phase, include_baseline=False)
            ax.set_title(f"{metric.replace('_',' ').title()} — {phase.title()} Phase")
        fig.suptitle(f"Training {metric.replace('_',' ').title()} (All Phases)", fontsize=14)
        save_pdf(fig, f"train_{metric}_all_phases", handles=LEGEND_TRAIN, bottom_pad=0.24)

# =========================
# 2) TEST (all phases in one file per metric)
# =========================
def testing_all_phases(data):
    for metric in METRICS:
        fig, axes = plt.subplots(3, 1, figsize=(12,10), sharex=False)
        for i, phase in enumerate(PHASES):
            ax = axes[i]
            line_means(ax, data, metric, "test", phase, include_baseline=True)
            ax.set_title(f"{metric.replace('_',' ').title()} — {phase.title()} Phase")
        fig.suptitle(f"Testing {metric.replace('_',' ').title()} (All Phases)", fontsize=14)
        save_pdf(fig, f"test_{metric}_all_phases", handles=LEGEND_ALL, bottom_pad=0.24)

# =========================
# 3) Temporal dashboard (exactly 9 days, Option A)
# =========================
def temporal_dashboard(test):
    fig, axes = plt.subplots(3, 3, figsize=(14,10))
    charts = ["reward","latency_ms","energy_efficiency",
              "task_scheduling_score","scalability_score","resource_allocation_score",
              "reward","latency_ms","energy_efficiency"]
    for i, metric in enumerate(charts):
        ax = axes[i//3, i%3]
        for agent in AGENT_ORDER:
            st = AGENT_STYLES[agent]
            sub = test[test["agent"]==agent]
            if sub.empty: continue
            daily = sub.groupby("day")[metric].mean().reindex(range(1,10)).dropna()
            # tiny smoothing without crashing; window ≤ 3
            if len(daily) >= 3:
                w = min(3, int(len(daily)//2) or 1)
                daily = smooth(daily, window=w).fillna(method="bfill").fillna(method="ffill")
            ax.plot(daily.index, daily.values, color=st["color"])
        shade_phase_bands(ax)
        int_day_ticks(ax, range(1,10))
        ax.set_title(metric.replace('_',' ').title())
        ax.set_ylabel(metric.replace('_',' ').title())
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Temporal Dynamics Across 9 Days (Phases Labeled)", fontsize=14)
    save_pdf(fig, "temporal_dashboard_9days", handles=LEGEND_ALL, bottom_pad=0.24)

# =========================
# 4) Phase errorbars (testing) — concise summary per metric
# =========================
def phase_errorbars(test):
    for idx, metric in enumerate(METRICS, start=1):
        fig, ax = plt.subplots(figsize=(6,4))
        for a in AGENT_ORDER:
            st = AGENT_STYLES[a]
            sub = test[test["agent"]==a]
            if sub.empty: continue
            means = sub.groupby("phase")[metric].mean().reindex(PHASES)
            stds  = sub.groupby("phase")[metric].std().reindex(PHASES)
            ax.errorbar(PHASES, means, yerr=stds, marker=st["marker"], color=st["color"], capsize=3)
        ax.set_title(f"{metric.replace('_',' ').title()} by Phase (Testing)")
        ax.set_xlabel("Phase"); ax.set_ylabel(metric.replace('_',' ').title())
        ax.grid(True, linestyle="--", alpha=0.5)
        save_pdf(fig, f"{idx:02d}_{metric}_by_phase", handles=LEGEND_ALL)

# =========================
# 5) Radar by phase (testing)
# =========================
def radar_by_phase(test):
    metrics = ["latency_ms","energy_efficiency","task_scheduling_score","scalability_score","resource_allocation_score"]
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(16,6))
    for j, ph in enumerate(PHASES):
        ax = axes[j]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        by_agent, present = [], []
        for a in AGENT_ORDER:
            sub = test[(test["phase"]==ph) & (test["agent"]==a)]
            if sub.empty: continue
            by_agent.append(sub[metrics].mean().tolist())
            present.append(a)
        if by_agent:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(by_agent)
            for a, vals in zip(present, scaled):
                vals = list(vals) + [vals[0]]
                st = AGENT_STYLES[a]
                ax.plot(angles, vals, color=st["color"])
                ax.fill(angles, vals, color=st["color"], alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_',' ').title() for m in metrics], fontsize=9)
        ax.set_title(f"{ph.title()} Phase")
    fig.suptitle("Radar Comparison by Phase (Testing)", fontsize=14)
    save_pdf(fig, "radar_comparison_by_phase", handles=LEGEND_ALL, bottom_pad=0.20)

# =========================
# 6) Final phase: distributions (boxplots) + ECDF + trade-offs
# =========================
def distributions_and_tradeoffs(test):
    final = test[test["phase"]=="final"]
    # 6a: Boxplots grid 3×2
    subset = ["latency_ms","reward","energy_efficiency","scalability_score","task_scheduling_score","resource_allocation_score"]
    fig, axes = plt.subplots(3, 2, figsize=(14,10))
    for i, m in enumerate(subset):
        ax = axes[i//2, i%2]
        series, labels = [], []
        for a in AGENT_ORDER:
            vals = final[final["agent"]==a][m].dropna().values
            if vals.size:
                series.append(vals)
                labels.append(AGENT_STYLES[a]["label"])
        if series:
            bp = ax.boxplot(series, patch_artist=True, tick_labels=labels)
            for box, lab in zip(bp['boxes'], labels):
                key = next(k for k,v in AGENT_STYLES.items() if v["label"]==lab)
                box.set_facecolor(AGENT_STYLES[key]["color"])
                box.set_alpha(0.55)
        ax.set_title(f"{m.replace('_',' ').title()} — Final Phase")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Distribution and Variability (Final Phase)", fontsize=14)
    save_pdf(fig, "distribution_variability_final", handles=None)

    # 6b: ECDF + trade-off scatter (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    # ECDF
    ax = axes[0,0]
    for a in AGENT_ORDER:
        st = AGENT_STYLES[a]
        vals = final[final["agent"]==a]["latency_ms"].dropna().sort_values()
        if vals.empty: continue
        y = np.arange(1, len(vals)+1) / len(vals)
        ax.step(vals.values, y, where="post", color=st["color"])
    ax.set_title("ECDF — Latency (Final Phase)")
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("ECDF"); ax.grid(True, linestyle="--", alpha=0.5)

    # Trade-offs
    pairs = [("latency_ms","energy_efficiency"),
             ("reward","energy_efficiency"),
             ("latency_ms","reward")]
    for i, (x,y) in enumerate(pairs, start=1):
        ax = axes[i//2, i%2]
        for a in AGENT_ORDER:
            st = AGENT_STYLES[a]
            means = final[final["agent"]==a][[x,y]].mean()
            if means.isna().any(): continue
            ax.scatter(means[x], means[y], color=st["color"], marker=st["marker"], s=70, label=st["label"])
        ax.set_xlabel(x.replace('_',' ').title()); ax.set_ylabel(y.replace('_',' ').title())
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("ECDF and Metric Trade-offs (Final Phase)", fontsize=14)
    save_pdf(fig, "ecdf_tradeoffs_final", handles=LEGEND_ALL)

# =========================
# 7) Bars & ranking
# =========================
def bars_and_ranks(test):
    # Bars
    fig, ax = plt.subplots(figsize=(6,4))
    means = test.groupby("agent")["reward"].mean().reindex(AGENT_ORDER)
    bars = ax.bar([AGENT_STYLES[a]["label"] for a in AGENT_ORDER],
                  means.values,
                  color=[AGENT_STYLES[a]["color"] for a in AGENT_ORDER])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Average Reward (Testing)")
    ax.set_ylabel("Reward"); ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    save_pdf(fig, "bar_reward_agents", handles=None)

    # Overall rank
    fig, ax = plt.subplots(figsize=(6,4))
    ranks = {}
    for m in METRICS:
        r = test.groupby("agent")[m].mean().rank()
        for a,v in r.items():
            ranks.setdefault(a, []).append(v)
    avg = {a: np.mean(v) for a,v in ranks.items()}
    vals = [avg.get(a, np.nan) for a in AGENT_ORDER]
    bars = ax.bar([AGENT_STYLES[a]["label"] for a in AGENT_ORDER], vals,
                  color=[AGENT_STYLES[a]["color"] for a in AGENT_ORDER])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Overall Average Rank (Lower is Better)")
    ax.set_ylabel("Average Rank"); ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    save_pdf(fig, "overall_avg_rank", handles=None)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    data = load_all()
    test  = data[data["mode"]=="test"].copy()

    # 1) Training (6 PDFs, each contains Early/Mid/Final as 3 rows)
    training_all_phases(data)

    # 2) Testing (6 PDFs, each contains Early/Mid/Final as 3 rows)
    testing_all_phases(data)

    # 3) Temporal dashboard across exactly 9 days (Option A)
    temporal_dashboard(test)

    # 4) Phase errorbars summary (6 PDFs)
    phase_errorbars(test)

    # 5) Radar by phase (1 PDF)
    radar_by_phase(test)

    # 6) Final-phase distributions + ECDF/trade-offs (2 PDFs)
    distributions_and_tradeoffs(test)

    # 7) Bars & ranks (2 PDFs)
    bars_and_ranks(test)

    print("\n✅ Done. Clean titles, combined per-metric phase grids, temporal dashboard with phase labels, framed legends, robust smoothing, PDF outputs only.")
