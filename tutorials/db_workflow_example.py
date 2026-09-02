"""
db_workflow_example.py - End-to-end eggp DB mode workflow
==========================================================

This tutorial shows the full eggp DB-backed workflow: GP search with fitness
caching, persistence across sessions, and Pareto front extraction with
confidence intervals.

The DB mode enables:
  1. **Fitness caching**: expressions evaluated in a previous run are recognised
     and their cached fitness is reused (skipping NLopt optimisation).
  2. **Persistence**: the DB accumulates results over multiple sessions; a new
     run picks up where the last left off.
  3. **CI reporting**: the Pareto front includes profile-likelihood confidence
     intervals for each parameter.

Run from the ``tutorials/`` directory::

    python db_workflow_example.py
"""

import os
import time
import sqlite3

import numpy as np
import pandas as pd

from eggp import EGGP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def db_stats(db_path):
    """Print DB statistics."""
    if not os.path.exists(db_path):
        print(f"  [DB not found: {db_path}]")
        return {}
    con = sqlite3.connect(db_path)
    stats = {}
    for table in ["dataset_fit", "eclass", "enode"]:
        try:
            stats[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            stats[table] = 0
    # Count datasets
    try:
        stats["datasets"] = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    except sqlite3.OperationalError:
        stats["datasets"] = 0
    con.close()
    print(f"  dataset_fit: {stats['dataset_fit']} rows | "
          f"eclass: {stats['eclass']} | "
          f"datasets: {stats['datasets']}")
    return stats


def make_dataset(path, rng_seed=42, n=200):
    """Create a synthetic dataset."""
    rng = np.random.default_rng(rng_seed)
    x0 = rng.uniform(-3, 3, n)
    x1 = rng.uniform(-3, 3, n)
    y = np.sin(x0) + 0.5 * x1
    df = pd.DataFrame({"x0": x0, "x1": x1, "y": y})
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_csv = "db_workflow_data.csv"
    egraph_db = "db_workflow_egraph.db"
    fit_db = "db_workflow_fit.db"

    for f in (data_csv, egraph_db, fit_db):
        if os.path.exists(f):
            os.remove(f)

    df = make_dataset(data_csv)
    X = df[["x0", "x1"]]
    y = df["y"]

    # ===================================================================
    # Session 1: first run — builds the DB from scratch
    # ===================================================================
    print("=" * 64)
    print("Session 1: first run (no prior DB)")
    print("=" * 64)

    model1 = EGGP(
        gen=10,
        nPop=50,
        maxSize=12,
        nTournament=3,
        pc=0.8,
        pm=0.3,
        nonterminals="add,sub,mul,div,sin",
        loss="MSE",
        optIter=30,
        optRepeat=2,
        nParams=1,
        folds=1,
        simplify=True,
        max_time=30,
        # --- DB parameters ---
        dbFile=egraph_db,
        dbFitFile=fit_db,
        dbDataset="demo",
        dbFlushEvery=5,
    )

    t0 = time.time()
    model1.fit(X, y)
    elapsed1 = time.time() - t0

    print(f"\n  Finished in {elapsed1:.1f}s")
    print(f"  Pareto front ({len(model1.results)} expressions):")
    print(model1.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    print("  DB stats:")
    db_stats(fit_db)

    # ===================================================================
    # Session 2: resume — reuses cached fitness from Session 1
    # ===================================================================
    print("\n" + "=" * 64)
    print("Session 2: resume from existing DB (fitness reuse)")
    print("=" * 64)

    model2 = EGGP(
        gen=10,
        nPop=50,
        maxSize=12,
        nTournament=3,
        pc=0.8,
        pm=0.3,
        nonterminals="add,sub,mul,div,sin",
        loss="MSE",
        optIter=30,
        optRepeat=2,
        nParams=1,
        folds=1,
        simplify=True,
        max_time=30,
        # --- same DB files ---
        dbFile=egraph_db,
        dbFitFile=fit_db,
        dbDataset="demo",
        dbFlushEvery=5,
    )

    t0 = time.time()
    model2.fit(X, y)
    elapsed2 = time.time() - t0

    print(f"\n  Finished in {elapsed2:.1f}s")
    print(f"  Pareto front ({len(model2.results)} expressions):")
    print(model2.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    print("  DB stats:")
    db_stats(fit_db)

    # ===================================================================
    # Session 3: continue searching — the DB keeps growing
    # ===================================================================
    print("\n" + "=" * 64)
    print("Session 3: continue searching (more generations)")
    print("=" * 64)

    model3 = EGGP(
        gen=20,
        nPop=50,
        maxSize=12,
        nTournament=3,
        pc=0.8,
        pm=0.3,
        nonterminals="add,sub,mul,div,sin",
        loss="MSE",
        optIter=30,
        optRepeat=2,
        nParams=1,
        folds=1,
        simplify=True,
        max_time=60,
        dbFile=egraph_db,
        dbFitFile=fit_db,
        dbDataset="demo",
        dbFlushEvery=10,
    )

    t0 = time.time()
    model3.fit(X, y)
    elapsed3 = time.time() - t0

    print(f"\n  Finished in {elapsed3:.1f}s")
    print(f"  Pareto front ({len(model3.results)} expressions):")
    print(model3.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    print("  DB stats:")
    db_stats(fit_db)

    # ===================================================================
    # Final: extract Pareto front with CI columns
    # ===================================================================
    print("\n" + "=" * 64)
    print("Final: Pareto front with profile-likelihood CIs")
    print("=" * 64)

    model_final = EGGP(
        gen=5,
        nPop=50,
        maxSize=12,
        nTournament=3,
        pc=0.8,
        pm=0.3,
        nonterminals="add,sub,mul,div,sin",
        loss="MSE",
        optIter=30,
        optRepeat=2,
        nParams=1,
        folds=1,
        simplify=True,
        max_time=15,
        dbFile=egraph_db,
        dbFitFile=fit_db,
        dbDataset="demo",
        dbFlushEvery=0,
    )
    model_final.fit(X, y)

    cols = ["Expression", "loss_train", "size"]
    ci_cols = [c for c in model_final.results.columns if c.startswith("ci_")]
    if ci_cols:
        cols.extend(ci_cols)

    print(f"\nPareto front ({len(model_final.results)} expressions):")
    print(model_final.results[cols].to_string(index=False))

    if ci_cols:
        print(f"\nCI columns: {', '.join(ci_cols)}")
        print("Each parameter has lower/upper bounds (95% profile-likelihood CI).")

    # Export to CSV
    out_csv = "db_workflow_results.csv"
    model_final.results[cols].to_csv(out_csv, index=False)
    print(f"\nResults exported to {out_csv}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 64)
    print("DB mode workflow summary")
    print("=" * 64)
    print(f"  Session 1 (fresh):     {elapsed1:.1f}s")
    print(f"  Session 2 (resume):    {elapsed2:.1f}s  (reuses cached fitness)")
    print(f"  Session 3 (continue):  {elapsed3:.1f}s  (DB accumulates)")
    print(f"  Final DB stats:")
    db_stats(fit_db)
    print()
    print("  The DB persists across runs.  A new EGGP object pointed at the")
    print("  same dbFile/dbFitFile will load the fitness cache and skip")
    print("  re-evaluation of expressions that were already fitted.")
    print("=" * 64)

    # Cleanup
    for f in (data_csv, egraph_db, fit_db, out_csv):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
