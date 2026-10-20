# P4 density-ratio rarity scoring: KILLED (2026-10-20)

## Stop rule
The rule was set before any results: if the v1 score fails, one fix is allowed. If the top tail is still not dominated by genuinely interesting scenarios after that fix, the idea is dead.

## v1 (scripts/score.py, results/scores.csv)
- Per-scenario diagonal Gaussian scored against a 64-component full-covariance GMM fit to the whole corpus. Scenario score = mean log p(x|s) - log q(x).
- Top tail: sparse, smooth, uninteresting scenes. 0% have a stationary ego, median 24 tracks.
- Bottom tail: 98% have a stationary ego, against 16% across the corpus.
- Neither tail caught the rare artifacts (position jumps, speeds over 40 m/s).
- All scores came out strongly negative, which points to a mismatch between the two model families.

## v2, the one fix (scripts/score_v2.py, results/scores_v2.csv)
- Agent-level scoring on 9 continuous dimensions, moving steps only.
- Scenario score = max agent score minus the scenario's median agent score.
- Hand labels for the top 50: 45 ARTIFACT, 5 BORING, 0 INTERESTING (see results/v2_top50_labels.csv).
- The artifacts are mostly acceleration, jerk, and heading noise in the tracking, often on agents far from ego (up to 236 m).

## Caveat
The v2 primary score was effectively -log q(x). The reference model p0 (fit on agents within 30 m of ego) was fit but never used in that score.
The alternative r_alt = log p0 - log q was computed but never hand-labeled. It barely correlates with the primary score (Spearman 0.023), and the two share none of their top 20.
