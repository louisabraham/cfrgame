"""Multiple Oracle (multiple_oracle.py) for the n-player CfR game with
friction, tau = 0.03, P = 1, rho = 0 (independent failures).

    python bench/scripts/multiplayer.py <n>   ->  bench/multiplayer/n<n>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import multiple_oracle as mo  # noqa: E402

n, tau = int(sys.argv[1]), 0.03
atoms, x, info = mo.multiple_oracle(n, tau, 1.0, eps=1e-10, max_iter=1000, time_limit=400)
check = mo.exploitability(atoms, x, n, tau, 1.0)
k = x > 1e-9
s = np.sort(atoms[k])
groups = np.split(s, np.nonzero(np.diff(s) > 2e-3)[0] + 1)  # atoms: points < 2e-3 apart
w = [float(x[k][np.isin(atoms[k], g)].sum()) for g in groups]
out = {"n": n, "tau": tau, "iterations": info["iterations"], "time": info["time"],
       "nashconv": info["nashconv"], "nashconv_check": check, "X": len(atoms),
       "support": int(k.sum()), "atoms": [float(g.mean()) for g in groups], "weights": w,
       "mean": float(atoms @ x), "value": info["value"]}
(ROOT / "bench" / "multiplayer" / f"n{n}.json").write_text(json.dumps(out, indent=1))
print(n, "done", time.strftime("%H:%M:%S"))
