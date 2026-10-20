"""P4 Block 1: per-agent-step kinematic features + scenario meta for AV2 MF val."""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from scipy.signal import savgol_filter
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet

ROOT = Path("data/av2_mf/val")
TYPES = ["vehicle", "pedestrian", "cyclist", "motorcyclist", "bus"]
DT = 0.1
W = 5  # Savitzky-Golay window

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def sg(x, deriv, poly=2):
    return savgol_filter(x, W, poly, deriv=deriv, delta=DT, axis=0, mode="interp")

def process(d):
    sc = load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
    ego = next(t for t in sc.tracks if t.track_id == "AV")
    ego_ts = np.array([s.timestep for s in ego.object_states])
    ego_pos_full = np.full((110, 2), np.nan); ego_pos_full[ego_ts] = [s.position for s in ego.object_states]
    ego_vel_full = np.full((110, 2), np.nan); ego_vel_full[ego_ts] = np.stack([sg(ego_pos_full[ego_ts], 1)]) if len(ego_ts) >= W else 0
    ego_hd_full = np.full(110, np.nan); ego_hd_full[ego_ts] = [s.heading for s in ego.object_states]
    epos = ego_pos_full[ego_ts]
    ego_path = float(np.linalg.norm(np.diff(epos, axis=0), axis=1).sum())
    ego_max_speed = float(np.nanmax(np.linalg.norm(ego_vel_full, axis=1)))
    rows = []
    max_speed_any = max_accel_any = max_hjump = 0.0
    n_posjump = n_missing = 0
    for t in sc.tracks:
        st = t.object_states
        ts = np.array([s.timestep for s in st])
        pos = np.array([s.position for s in st], dtype=float)
        hd = np.array([s.heading for s in st], dtype=float)
        if len(ts) > 1:
            n_posjump += int((np.linalg.norm(np.diff(pos, axis=0), axis=1) > 5).sum())
            max_hjump = max(max_hjump, float(np.degrees(np.abs(wrap(np.diff(hd)))).max()))
        if len(ts) and (ts[-1] - ts[0] + 1) != len(ts):
            n_missing += 1
        otype = t.object_type.value
        if otype not in TYPES or len(ts) < 20:
            continue
        v = sg(pos, 1); a = sg(pos, 2); j = sg(pos, 3, poly=3)
        speed = np.linalg.norm(v, axis=1)
        u = v / np.maximum(speed, 1e-3)[:, None]
        # direction: velocity unit vector; fall back to heading when nearly stopped
        slow = speed < 0.5
        u[slow] = np.c_[np.cos(hd[slow]), np.sin(hd[slow])]
        n = np.c_[-u[:, 1], u[:, 0]]
        long_a = (a * u).sum(1); lat_a = (a * n).sum(1)
        jerk = (j * u).sum(1)
        yaw_rate = sg(np.unwrap(hd), 1)
        rel = pos - ego_pos_full[ts]
        dist = np.linalg.norm(rel, axis=1)
        rel_speed = np.linalg.norm(v - ego_vel_full[ts], axis=1)
        rel_hd = wrap(hd - ego_hd_full[ts])
        max_speed_any = max(max_speed_any, float(speed.max()))
        max_accel_any = max(max_accel_any, float(np.linalg.norm(a, axis=1).max()))
        df = pd.DataFrame({"speed": speed, "long_accel": long_a, "lat_accel": lat_a, "yaw_rate": yaw_rate,
                           "jerk": jerk, "dist_to_ego": dist, "rel_speed_to_ego": rel_speed,
                           "rel_heading_to_ego": rel_hd}).astype("float32")
        for ty in TYPES:
            df[f"type_{ty}"] = np.float32(otype == ty)
        df.insert(0, "timestep", ts.astype("int16"))
        df.insert(0, "track_id", t.track_id)
        df.insert(0, "scenario_id", sc.scenario_id)
        rows.append(df)
    meta = dict(scenario_id=sc.scenario_id, city=sc.city_name, num_tracks=len(sc.tracks), ego_path_m=ego_path,
                ego_max_speed=ego_max_speed, ego_stationary=ego_path < 1.0, max_speed_any=max_speed_any,
                max_accel_any=max_accel_any, max_heading_jump_deg=max_hjump, n_pos_jumps_gt5m=n_posjump,
                frac_tracks_missing_steps=n_missing / len(sc.tracks))
    return (pd.concat(rows, ignore_index=True) if rows else None), meta

def safe(d):
    try:
        return process(d)
    except Exception as e:
        return None, dict(scenario_id=d.name, error=repr(e))

if __name__ == "__main__":
    t0 = time.time()
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if len(sys.argv) > 1: dirs = dirs[: int(sys.argv[1])]
    Path("features").mkdir(exist_ok=True)
    writer, metas, buf, nrows = None, [], [], 0
    with Pool(8) as pool:
        for i, (df, m) in enumerate(pool.imap(safe, dirs, chunksize=16)):
            metas.append(m)
            if df is not None: buf.append(df)
            if len(buf) >= 200 or (i == len(dirs) - 1 and buf):
                tab = pa.Table.from_pandas(pd.concat(buf, ignore_index=True), preserve_index=False)
                writer = writer or pq.ParquetWriter("features/agent_steps.parquet", tab.schema)
                writer.write_table(tab); nrows += tab.num_rows; buf = []
            if i % 2000 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
    writer.close()
    meta = pd.DataFrame(metas); meta.to_parquet("features/scenario_meta.parquet", index=False)
    print("rows", nrows, "scenarios", len(meta), "errors", meta.get("error", pd.Series()).notna().sum(), f"{time.time()-t0:.0f}s")
