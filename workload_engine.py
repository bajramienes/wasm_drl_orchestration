import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class PhaseParams:
    lam: float            # Poisson arrivals (req/s)
    mean_burst: float     # geometric mean burst size
    duration_s: int       # per-episode duration
    mix: dict             # cpu/io/mixed percentage

def geometric_burst(rng: np.random.Generator, mean: float) -> int:
    # geometric with mean = (1-p)/p -> p = 1/(mean+1)
    p = 1.0 / (mean + 1.0)
    return int(rng.geometric(p))

def generate_trace(seed: int, phase: str, params: PhaseParams, out_dir: Path, episodes: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    traces = []
    for ep in range(1, episodes + 1):
        # arrivals over duration
        n = rng.poisson(params.lam * params.duration_s)
        bursts = []
        remaining = n
        while remaining > 0:
            b = min(remaining, max(1, geometric_burst(rng, params.mean_burst)))
            bursts.append(b)
            remaining -= b

        # assign each request a type according to mix
        types = rng.choice(
            ["cpu","io","mixed"],
            size=n,
            p=[params.mix["cpu"]/100, params.mix["io"]/100, params.mix["mixed"]/100]
        ).tolist()

        traces.append({
            "episode": ep,
            "phase": phase,
            "seed": seed,
            "duration_s": params.duration_s,
            "arrivals": int(n),
            "bursts": bursts,
            "types": types
        })

    fp = out_dir / f"{phase}_seed{seed}_trace.json"
    fp.write_text(json.dumps(traces, indent=2))
    return fp

def load_trace(fp: str):
    return json.loads(Path(fp).read_text())
