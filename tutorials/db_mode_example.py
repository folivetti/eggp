"""
12 - eggp in DB mode: persistent e-graph with fitness caching
=============================================================

This tutorial shows the *DB-backed* eggp mode: instead of keeping everything
in memory, eggp stores evaluated expressions and their fitness in a SQLite
database.  This enables:

1. **Fitness reuse**: expressions evaluated in a previous run are recognised
   and their cached fitness is reused (skipping the expensive NLopt
   optimisation).
2. **Expression deduplication**: crossover and mutation check the DB for
   already-explored expressions, avoiding re-exploration of the search space.
3. **Persistence across runs**: the DB accumulates results over multiple
   sessions; a new run picks up where the last left off.

The in-memory GP loop is unchanged when ``dbFile=""`` (the default), so there
is zero overhead when the DB mode is not used.

Run from the ``tutorials/`` directory::

    python 12_db_mode_example.py
"""

import os
import time
import sqlite3

import numpy as np
import pandas as pd

from eggp import EGGP

# ---------------------------------------------------------------------------
# Helper: inspect the DB
# ---------------------------------------------------------------------------

def db_stats(db_path):
    """Print basic statistics about the eggp database."""
    if not os.path.exists(db_path):
        print(f"  [DB not found: {db_path}]")
        return
    con = sqlite3.connect(db_path)
    n_fit = con.execute("SELECT COUNT(*) FROM dataset_fit").fetchone()[0]
    n_ds  = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    n_ec  = con.execute("SELECT COUNT(*) FROM eclass").fetchone()[0] if _table_exists(con, "eclass") else 0
    con.close()
    print(f"  dataset_fit rows: {n_fit}  |  datasets: {n_ds}  |  e-classes: {n_ec}")

def _table_exists(con, name):
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_dataset(path):
    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.uniform(-3, 3, n)
    x1 = rng.uniform(-3, 3, n)
    y  = np.sin(x0) + 0.5 * x1
    df = pd.DataFrame({"x0": x0, "x1": x1, "y": y})
    df.to_csv(path, index=False)
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_csv = "db_tutorial_data.csv"
    db_file  = "db_tutorial.db"

    # Clean up any previous run
    for f in (data_csv, db_file):
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
        dbFile=db_file,
        dbDataset="tutorial",
        dbFlushEvery=5,        # flush to DB every 5 generations
    )

    t0 = time.time()
    model1.fit(X, y)
    elapsed1 = time.time() - t0

    print(f"\n  Finished in {elapsed1:.1f}s")
    print(f"  Pareto front ({len(model1.results)} expressions):")
    print(model1.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    db_stats(db_file)

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
        # --- same DB file ---
        dbFile=db_file,
        dbDataset="tutorial",
        dbFlushEvery=5,
    )

    t0 = time.time()
    model2.fit(X, y)
    elapsed2 = time.time() - t0

    print(f"\n  Finished in {elapsed2:.1f}s")
    print(f"  Pareto front ({len(model2.results)} expressions):")
    print(model2.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    db_stats(db_file)

    # ===================================================================
    # Session 3: even more generations — the DB keeps growing
    # ===================================================================
    print("\n" + "=" * 64)
    print("Session 3: continue searching (DB accumulates)")
    print("=" * 64)

    model3 = EGGP(
        gen=20,                  # more generations this time
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
        dbFile=db_file,
        dbDataset="tutorial",
        dbFlushEvery=10,
    )

    t0 = time.time()
    model3.fit(X, y)
    elapsed3 = time.time() - t0

    print(f"\n  Finished in {elapsed3:.1f}s")
    print(f"  Pareto front ({len(model3.results)} expressions):")
    print(model3.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    db_stats(db_file)

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 64)
    print("DB mode summary")
    print("=" * 64)
    print(f"  Session 1 (fresh):     {elapsed1:.1f}s")
    print(f"  Session 2 (resume):    {elapsed2:.1f}s  (reuses cached fitness)")
    print(f"  Session 3 (continue):  {elapsed3:.1f}s")
    print(f"  Final DB stats:")
    db_stats(db_file)
    print()
    print("  The DB file persists across runs.  A new EGGP object pointed at the")
    print("  same dbFile will load the fitness cache and skip re-evaluation of")
    print("  expressions that were already fitted.")
    print("=" * 64)

    # cleanup
    for f in (data_csv, db_file):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
