"""P4 Block 0 smoke test: load N AV2 motion-forecasting scenarios and print basic stats."""
import sys
from pathlib import Path
import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/av2_mf/val")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
dirs = sorted(d for d in root.iterdir() if d.is_dir())[:n]
print("| scenario_id | city | num_tracks | focal_len (steps) | ego_path_m | ego_max_speed_mps |")
print("|---|---|---|---|---|---|")
for d in dirs:
    sc = load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
    tracks = {t.track_id: t for t in sc.tracks}
    focal = tracks[sc.focal_track_id]
    ego = tracks["AV"]
    pos = np.array([s.position for s in ego.object_states])
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    vmax = float(max(np.linalg.norm(s.velocity) for s in ego.object_states))
    print(f"| {sc.scenario_id} | {sc.city_name} | {len(sc.tracks)} | {len(focal.object_states)} | {path:.1f} | {vmax:.2f} |")
