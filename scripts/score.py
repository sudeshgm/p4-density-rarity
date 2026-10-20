"""P4 Block 1: density-ratio rarity score + marginal-density baseline."""
import time, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from scipy.stats import spearmanr

FEATS = ["speed", "long_accel", "lat_accel", "yaw_rate", "jerk", "dist_to_ego", "rel_speed_to_ego",
         "rel_heading_to_ego", "type_vehicle", "type_pedestrian", "type_cyclist", "type_motorcyclist", "type_bus"]
SHRINK, NSUB, SEED = 0.2, 2_000_000, 0
LOG2PI = np.log(2 * np.pi)

t0 = time.time()
pf = pq.ParquetFile("features/agent_steps.parquet")
N = pf.metadata.num_rows
rng = np.random.default_rng(SEED)
pick = np.sort(rng.choice(N, size=min(NSUB, N), replace=False))
s1 = np.zeros(len(FEATS)); s2 = np.zeros(len(FEATS)); sub = []; off = 0
for g in range(pf.num_row_groups):
    X = pf.read_row_group(g, columns=FEATS).to_pandas().to_numpy(np.float64)
    s1 += X.sum(0); s2 += (X ** 2).sum(0)
    lo, hi = np.searchsorted(pick, [off, off + len(X)])
    sub.append(X[pick[lo:hi] - off]); off += len(X)
mu = s1 / N; sd = np.sqrt(s2 / N - mu ** 2)
Xs = (np.concatenate(sub) - mu) / sd
print(f"pass1 {time.time()-t0:.0f}s rows={N} sub={len(Xs)}", flush=True)
t1 = time.time()
gmm = GaussianMixture(n_components=64, covariance_type="full", random_state=SEED).fit(Xs)
print(f"gmm fit {time.time()-t1:.0f}s converged={gmm.converged_} iters={gmm.n_iter_}", flush=True)
del Xs, sub
t2 = time.time()
gvar = np.ones(len(FEATS))  # global variance in standardized space
out = []
for g in range(pf.num_row_groups):
    T = pf.read_row_group(g, columns=["scenario_id"] + FEATS).to_pandas()
    X = (T[FEATS].to_numpy(np.float64) - mu) / sd
    logq = gmm.score_samples(X)
    for sid, idx in T.groupby("scenario_id", sort=False).indices.items():
        x = X[idx]
        m = x.mean(0); v = (1 - SHRINK) * x.var(0) + SHRINK * gvar
        logp = -0.5 * (((x - m) ** 2 / v).sum(1) + np.log(v).sum() + len(FEATS) * LOG2PI)
        r = logp - logq[idx]
        out.append((sid, r.mean(), np.percentile(-logq[idx], 95), len(idx)))
print(f"scoring {time.time()-t2:.0f}s", flush=True)
S = pd.DataFrame(out, columns=["scenario_id", "density_ratio_score", "baseline_nll_p95", "n_agent_steps"])
meta = pd.read_parquet("features/scenario_meta.parquet")
S = meta.merge(S, on="scenario_id", how="left").sort_values("density_ratio_score", ascending=False)
S.to_csv("results/scores.csv", index=False)
rho = spearmanr(S.density_ratio_score, S.baseline_nll_p95, nan_policy="omit")
json.dump(dict(spearman=rho.statistic, gmm_converged=bool(gmm.converged_), gmm_iters=int(gmm.n_iter_),
               rows=int(N), total_s=time.time() - t0), open("results/score_summary.json", "w"), indent=1)
print("spearman", rho.statistic, f"total {time.time()-t0:.0f}s")
