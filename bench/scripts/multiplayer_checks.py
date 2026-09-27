"""Two checks for the n-player section of the report (tau = 0.03, P = 1,
rho = 0): the two-player Double Oracle on the same game (n = 2), and the
Multiple Oracle with pruning of zero-weight strategies (n = 2).

    python bench/scripts/multiplayer_checks.py  ->  bench/multiplayer/checks.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import baselines as bl  # noqa: E402
import multiple_oracle as mo  # noqa: E402

gp = dict(corr=0.0, noise=0.03, R=1, Z=0, P=1.0)
out = {}
for oracles in (1, 8):
    t = time.perf_counter()
    a, p, _, q, info = bl.double_oracle(gp, grid=None, log_points=12, time_limit=300, oracles=oracles)
    out[f"double_oracle_{oracles}"] = {"time": time.perf_counter() - t, "iterations": info["iterations"],
                                       "nashconv": mo.exploitability(a, (p + q) / 2, 2, 0.03, 1.0)}
atoms, x, info = mo.multiple_oracle(2, 0.03, 1.0, eps=1e-10, max_iter=100, prune=True)
out["multiple_oracle_pruned"] = {"iterations": info["iterations"], "nashconv": info["nashconv"],
                                 "X": len(atoms), "last_nashconv": [h[2] for h in info["history"][-6:]]}
(ROOT / "bench" / "multiplayer" / "checks.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
