# p4-density-rarity

P4: density/rarity analysis on Argoverse 2 Motion Forecasting (val split).

Setup: python3 -m venv .venv; pip install av2 pandas pyarrow numpy scikit-learn scipy matplotlib tqdm s5cmd
Data: s5cmd --no-sign-request cp 's3://argoverse/datasets/av2/motion-forecasting/val/*' data/av2_mf/val/
Smoke test: python scripts/smoke_test.py

## Status
Killed under the pre-registered stop rule. See [results/KILL_NOTE.md](results/KILL_NOTE.md).
