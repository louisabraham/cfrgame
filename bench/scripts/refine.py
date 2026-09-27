"""Double Oracle + Newton on atoms (hybrid.newton_atoms), from a Double Oracle
start or a fictitious-play start, for one tau. Single thread; P=1, rho=0.5.

    python bench/scripts/refine.py <tau> <do|fp>   ->  bench/refine/<start>_tau<tau>.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import baselines as bl  # noqa: E402
import hybrid as hy  # noqa: E402

tau, start = float(sys.argv[1]), sys.argv[2]
p, rho = 1.0, 0.5
gp = dict(corr=rho, noise=tau, R=1, Z=0, P=p)
t0 = time.perf_counter()
if start == "do":
    a, q1, _, q2, info = bl.double_oracle(gp, grid=None, log_points=12, time_limit=300)
    q = (q1 + q2) / 2
    sinfo = {"points": int((q > 1e-9).sum()), "iterations": info["iterations"]}
else:
    n = {0.01: 1024, 0.001: 2048, 0.0001: 4096}.get(tau, 1024)
    a, q, sinfo = hy.fp_start(p, tau, rho, n=n, time_limit=60)
    sinfo["n"] = n
t_start = time.perf_counter() - t0
qnc_start = hy.quasinashconv(p, tau, rho, a, q)
x, w, v, info = hy.newton_atoms(a, q, p, tau, rho, time_limit=600)
out = {"tau": tau, "start": start, "start_info": sinfo, "start_time": t_start,
       "start_qnc": qnc_start, "qnc": hy.quasinashconv(p, tau, rho, x, w), **info,
       "x": x.tolist(), "w": w.tolist()}
(ROOT / "bench" / "refine" / f"{start}_tau{tau:g}.json").write_text(json.dumps(out, indent=1))
print(tau, start, "done", time.strftime("%H:%M:%S"))
