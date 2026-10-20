"""P4 Block 2: agent-level density-ratio scoring (v2).
PRIMARY per-step r = -log q(x) (spec text, literal). SECONDARY r_alt = log p0(x) - log q(x),
p0 = GMM fit on steps with dist_to_ego <= 30 m. Spec is ambiguous about which was intended."""
import json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from sklearn.mixture import GaussianMixture
T = {}; t0 = time.time()
def lap(k):
    global t0; T[k] = round(time.time() - t0, 1); print(k, T[k], "s", flush=True); t0 = time.time()
CONT = ["speed","long_accel","lat_accel","yaw_rate","jerk","dist_to_ego","rel_speed_to_ego","rel_heading_to_ego"]
TYPES = ["type_vehicle","type_pedestrian","type_cyclist","type_motorcyclist","type_bus"]
tb = pq.read_table("features/agent_steps.parquet", columns=["scenario_id","track_id"] + CONT + TYPES)
n_all = tb.num_rows
tb = tb.filter(pc.greater_equal(tb["speed"], 0.5))
sid = tb["scenario_id"].combine_chunks().dictionary_encode()
tid = pd.Series(tb["track_id"].to_numpy(zero_copy_only=False))
key = pd.Series(sid.to_pandas(), dtype="category").astype(str) if False else None
sc_codes = sid.indices.to_numpy(); sc_names = np.array(sid.dictionary.to_pylist(), dtype=object)
cols = {c: tb[c].to_numpy().astype(np.float32) for c in CONT + TYPES}
del tb
h = cols.pop("rel_heading_to_ego")
X = np.column_stack([cols[c] for c in CONT[:-1]] + [np.sin(h), np.cos(h)]).astype(np.float32)
ok = np.isfinite(X).all(1)
X, sc_codes, tid = X[ok], sc_codes[ok], tid[ok].to_numpy()
otype = np.argmax(np.column_stack([cols[c] for c in TYPES]), 1)[ok]
dist = cols["dist_to_ego"][ok]; del cols
mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd; del X
print("rows all", n_all, "moving+finite", len(Z)); lap("load")
rng = np.random.default_rng(0)
def fit(idx, name):
    idx = rng.choice(idx, min(len(idx), 1_000_000), replace=False)
    reg = 1e-6
    while True:
        try:
            g = GaussianMixture(32, covariance_type="full", random_state=0, max_iter=200, reg_covar=reg).fit(Z[idx].astype(np.float64)); break
        except ValueError as e:
            print(name, "fit failed at reg_covar", reg, "->", str(e)[:80], flush=True); reg *= 10
    print(name, "reg_covar", reg, "n", len(idx), "converged", g.converged_, "n_iter", g.n_iter_, "lower_bound", g.lower_bound_, flush=True)
    return g, dict(n=int(len(idx)), converged=bool(g.converged_), n_iter=int(g.n_iter_), lower_bound=float(g.lower_bound_), reg_covar=reg)
q, iq = fit(np.arange(len(Z)), "q"); lap("fit_q")
p0, ip = fit(np.flatnonzero(dist <= 30), "p0"); lap("fit_p0")
lq = np.empty(len(Z), np.float32); lp = np.empty(len(Z), np.float32)
for i in range(0, len(Z), 2_000_000):
    lq[i:i+2_000_000] = q.score_samples(Z[i:i+2_000_000].astype(np.float64)); lp[i:i+2_000_000] = p0.score_samples(Z[i:i+2_000_000].astype(np.float64))
lap("score_steps")
df = pd.DataFrame({"sc": sc_codes, "tid": tid, "ot": otype, "r": -lq, "ra": lp - lq})
A = df.groupby(["sc","tid"], sort=False).agg(n=("r","size"), r=("r","mean"), ra=("ra","mean"), ot=("ot","first")).reset_index()
A = A[A.n >= 10]
TN = np.array(["vehicle","pedestrian","cyclist","motorcyclist","bus"])
def scen(col):
    g = A.groupby("sc")
    med = g[col].median(); cnt = g.size()
    im = g[col].idxmax()
    out = pd.DataFrame({"n_scored_agents": cnt, col+"_max": g[col].max(), col+"_median": med,
                        "agent": A.loc[im, "tid"].values, "type": TN[A.loc[im, "ot"].values]}, index=med.index)
    out["score"] = out[col+"_max"] - med
    return out
P, R = scen("r"), scen("ra")
S = pd.DataFrame({"scenario_id": sc_names[P.index], "n_scored_agents": P.n_scored_agents.values,
    "score_v2": P.score.values, "rarest_agent_id": P.agent.values, "rarest_agent_type": P.type.values,
    "agent_score_max": P.r_max.values, "agent_score_median": P.r_median.values,
    "score_v2_alt": R.score.values, "rarest_agent_id_alt": R.agent.values, "rarest_agent_type_alt": R.type.values})
meta = pd.read_parquet("features/scenario_meta.parquet")
v1 = pd.read_csv("results/scores.csv")[["scenario_id","density_ratio_score","baseline_nll_p95"]]
v1 = v1.groupby("scenario_id", as_index=False).mean()
S = meta.merge(S, on="scenario_id", how="left").merge(v1, on="scenario_id", how="left")
skip = S.n_scored_agents.fillna(0) < 3
S.loc[skip, ["score_v2","score_v2_alt"]] = np.nan
S["n_scored_agents"] = S.n_scored_agents.fillna(0).astype(int)
S.sort_values("score_v2", ascending=False).to_csv("results/scores_v2.csv", index=False); lap("aggregate")
info = dict(times=T, q=iq, p0=ip, n_scenarios=len(S), n_skipped_lt3_agents=int(skip.sum()),
            n_scored=int((~skip).sum()), n_moving_steps=int(len(Z)), n_scored_agents=int(len(A)))
json.dump(info, open("results/score_v2_info.json","w"), indent=1); print(info)
