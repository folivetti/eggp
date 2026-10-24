# Changelog for eggp

## 2.2.2

- **DB mode resumes the persisted e-graph**: when `--load-from` is not given,
  the stored graph is loaded with `loadGraphResident`, preserving e-class ids
  so the eid-keyed fitness cache stays valid across runs. Re-running with the
  same DB files no longer leaves stale `dataset_fit` rows that produce
  `<extraction failed>` in queries.
- **Fit-DB coherence**: stale `dataset_fit` rows (eids not in the graph) are
  pruned at startup.
- **`syncEGraphDB` surfaces `saveGraph` errors** (stderr + exception) instead
  of silently discarding them.
- Requires `srtree >= 3.0.0.6` and `srtree-db >= 0.1.3.3`.

## 2.2.1

- **3x speedup with multithreading** (-N4): parallel tree generation via
  `mapConcurrently` on e-graph snapshots; batch eqsat after all offspring
  inserted; fixed duplicate refitIds bug
- **eggp DB mode**: external SQLite database for fitness caching, expression
  deduplication, and periodic persistence (`--db-file`, `--db-dataset`,
  `--db-cache-size`, `--db-flush-every`)
- `crossoverDB`/`mutateDB` check both in-memory e-graph and DB for
  already-seen expressions
- Python API: `dbFile`, `dbDataset`, `dbCacheSize`, `dbFlushEvery` parameters
- SWIG binding updated for DB mode parameters
- Fixed profile-likelihood CI distribution mapping: non-NLL losses (MSE, LOG10,
  MAE, MAPE, Pinball) now correctly use `LeastSquares` distribution instead of
  defaulting to `Gaussian` (which adds a spurious sigma parameter)
- Changed profile method from `Constrained` to `Bates` for more robust CIs
- CI output uses `showNA` to display "NA" for NaN/Inf values
- Fixed `csvHeader` to use `actualMaxP` (computed from Pareto front) instead of
  `_nParams` arg, ensuring enough CI columns when `--n-params -1`
- Added `computeMaxP` to walk Pareto front and find actual max parameter count
- **`trace=True` no longer computes per-generation CIs**: `printExpr` now takes a
  `withCI` flag; the trace loop skips profile-likelihood CI computation per
  individual per generation (a severe slowdown), while the final Pareto front
  still computes CIs
- **`evaluate_best_model` picks the best-fit Pareto member**: selects the row
  with minimum `loss_train` (`_best_row`) instead of `results.iloc[-1]` (the
  largest/most complex member), fixing predictions on well-fit models
- Updated dependency: srtree >= 3.0.0.4, srtree-db >= 0.1.3.0

## 2.1.0

- Pareto front output now includes profile-likelihood confidence intervals
  (Constrained profile method) for all fitted parameters
- CSV output gains `t_lower,t_upper` columns per parameter

## 2.0.0

- Initial release with DB-backed GP mode
