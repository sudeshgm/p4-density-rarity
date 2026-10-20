"""P4 Block 1: BEV plots + contact sheets for top/bottom 100 by density-ratio score."""
import json, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet

ROOT = Path("data/av2_mf/val")

def bev(args):
    rank, sid, score, out = args
    d = ROOT / sid
    sc = load_argoverse_scenario_parquet(d / f"scenario_{sid}.parquet")
    m = json.load(open(next(d.glob("log_map_archive_*.json"))))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
    for ls in m["lane_segments"].values():
        for k in ("left_lane_boundary", "right_lane_boundary"):
            p = np.array([[q["x"], q["y"]] for q in ls[k]])
            ax.plot(p[:, 0], p[:, 1], color="#cccccc", lw=0.5, zorder=0)
    sm = None
    for t in sc.tracks:
        p = np.array([s.position for s in t.object_states])
        if t.track_id == "AV":
            ax.plot(p[:, 0], p[:, 1], "r-", lw=2, zorder=3); ego = p
        elif t.track_id == sc.focal_track_id:
            ax.plot(p[:, 0], p[:, 1], "b-", lw=2, zorder=3)
        else:
            v = np.linalg.norm([s.velocity for s in t.object_states], axis=1)
            sm = ax.scatter(p[:, 0], p[:, 1], c=v, cmap="Greys", vmin=0, vmax=20, s=2, zorder=2)
    c = ego.mean(0); ax.set_xlim(c[0] - 80, c[0] + 80); ax.set_ylim(c[1] - 80, c[1] + 80)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if sm is not None: fig.colorbar(sm, ax=ax, fraction=0.04, label="speed m/s")
    ax.set_title(f"#{rank} {sid[:8]} {sc.city_name} score={score:.2f}", fontsize=9)
    f = out / f"{rank:03d}_{sid}.png"; fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f

def sheets(files, out, name):
    paths = []
    for p in range(0, len(files), 25):
        ims = [Image.open(f).convert("RGB").resize((400, 400)) for f in files[p:p + 25]]
        sheet = Image.new("RGB", (2000, 2000), "white")
        for i, im in enumerate(ims): sheet.paste(im, ((i % 5) * 400, (i // 5) * 400))
        fp = out / f"contact_{name}_{p // 25 + 1}.png"; sheet.save(fp); paths.append(fp)
    return paths

if __name__ == "__main__":
    t0 = time.time()
    S = pd.read_csv("results/scores.csv").dropna(subset=["density_ratio_score"]).sort_values("density_ratio_score", ascending=False)
    for name, sel in [("top100", S.head(100)), ("bottom100", S.tail(100).iloc[::-1])]:
        out = Path("results") / name; out.mkdir(parents=True, exist_ok=True)
        jobs = [(i + 1, r.scenario_id, r.density_ratio_score, out) for i, r in enumerate(sel.itertuples())]
        with Pool(8) as pool: files = pool.map(bev, jobs)
        for p in sheets(files, Path("results"), name): print(p.resolve())
    print(f"inspect {time.time()-t0:.0f}s")
