"""
eggp_regression_integration.py - eggp and reggression integration via shared egraph
===================================================================================

This tutorial demonstrates cross-dataset analysis using a shared e-graph:

  1. Run eggp on dataset A — discovers expressions, stores in egraph + fit_a
  2. Run eggp on dataset B using the SAME egraph — adds new expressions, fit_b
  3. Query top-N from both fit databases via reggression
  4. Find the intersection: e-classes that are top-ranked on BOTH datasets

The shared egraph means expressions discovered on dataset A are available when
searching on dataset B. The intersection reveals "universally good" expressions
that perform well across different datasets.

Run from the ``tutorials/`` directory::

    python eggp_regression_integration.py
"""

import os
import time
import sqlite3

import numpy as np
import pandas as pd

from eggp import EGGP
from reggression import Reggression

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_dataset_a(path):
    """Dataset A: y = sin(x0) + 0.5*x1 (trigonometric)."""
    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.uniform(-3, 3, n)
    x1 = rng.uniform(-3, 3, n)
    y = np.sin(x0) + 0.5 * x1
    pd.DataFrame({"x0": x0, "x1": x1, "y": y}).to_csv(path, index=False)


def make_dataset_b(path):
    """Dataset B: y = x0*x1 + 0.3*x0 (multiplicative)."""
    rng = np.random.default_rng(99)
    n = 200
    x0 = rng.uniform(-3, 3, n)
    x1 = rng.uniform(-3, 3, n)
    y = x0 * x1 + 0.3 * x0
    pd.DataFrame({"x0": x0, "x1": x1, "y": y}).to_csv(path, index=False)


def db_stats(db_path, label=""):
    """Print DB statistics."""
    if not os.path.exists(db_path):
        print(f"  [{label} not found]")
        return
    con = sqlite3.connect(db_path)
    tables = {}
    for t in ["dataset_fit", "eclass", "enode"]:
        try:
            tables[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            tables[t] = 0
    con.close()
    print(f"  {label}: fit_rows={tables['dataset_fit']} "
          f"eclasses={tables['eclass']} enodes={tables['enode']}")


def get_top_ids(fit_db, dataset, n):
    """Get the top-N eclass IDs from a fit DB."""
    con = sqlite3.connect(fit_db)
    ds_rows = con.execute(
        "SELECT id FROM dataset WHERE name = ?", (dataset,)
    ).fetchall()
    if not ds_rows:
        con.close()
        return []
    dsid = ds_rows[0][0]
    rows = con.execute(
        "SELECT eid FROM dataset_fit "
        "WHERE dataset_id = ? AND fitted = 1 AND fitness IS NOT NULL "
        "ORDER BY fitness DESC LIMIT ?",
        (dsid, n)
    ).fetchall()
    con.close()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Files ---
    data_a_csv = "integ_data_a.csv"
    data_b_csv = "integ_data_b.csv"
    egraph_db = "integ_egraph.db"
    fit_a_db = "integ_fit_a.db"
    fit_b_db = "integ_fit_b.db"

    for f in (data_a_csv, data_b_csv, egraph_db, fit_a_db, fit_b_db):
        if os.path.exists(f):
            os.remove(f)

    # --- Create datasets ---
    make_dataset_a(data_a_csv)
    make_dataset_b(data_b_csv)
    df_a = pd.read_csv(data_a_csv)
    df_b = pd.read_csv(data_b_csv)

    # ===================================================================
    # Step 1: Run eggp on dataset A
    # ===================================================================
    print("=" * 64)
    print("Step 1: Run eggp on dataset A (trigonometric)")
    print("  y = sin(x0) + 0.5*x1")
    print("=" * 64)

    model_a = EGGP(
        gen=15,
        nPop=80,
        maxSize=12,
        nTournament=3,
        pc=0.8,
        pm=0.3,
        nonterminals="add,sub,mul,div,sin,cos",
        loss="MSE",
        optIter=30,
        optRepeat=2,
        nParams=1,
        folds=1,
        simplify=True,
        max_time=45,
        dbFile=egraph_db,
        dbFitFile=fit_a_db,
        dbDataset="A",
        dbFlushEvery=5,
    )

    t0 = time.time()
    model_a.fit(df_a[["x0", "x1"]], df_a["y"])
    elapsed_a = time.time() - t0

    print(f"\n  Finished in {elapsed_a:.1f}s")
    print(f"  Pareto front ({len(model_a.results)} expressions):")
    print(model_a.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    db_stats(egraph_db, "egraph.db (after A)")
    db_stats(fit_a_db, "fit_a.db")

    # ===================================================================
    # Step 2: Run eggp on dataset B using the SAME egraph
    # ===================================================================
    print("\n" + "=" * 64)
    print("Step 2: Run eggp on dataset B (multiplicative)")
    print("  y = x0*x1 + 0.3*x0")
    print("  (shares egraph.db with dataset A)")
    print("=" * 64)

    model_b = EGGP(
        gen=15,
        nPop=80,
        maxSize=12,
        nTournament=3,
        pc=0.8,
        pm=0.3,
        nonterminals="add,sub,mul,div,sin,cos",
        loss="MSE",
        optIter=30,
        optRepeat=2,
        nParams=1,
        folds=1,
        simplify=True,
        max_time=45,
        dbFile=egraph_db,
        dbFitFile=fit_b_db,
        dbDataset="B",
        dbFlushEvery=5,
    )

    t0 = time.time()
    model_b.fit(df_b[["x0", "x1"]], df_b["y"])
    elapsed_b = time.time() - t0

    print(f"\n  Finished in {elapsed_b:.1f}s")
    print(f"  Pareto front ({len(model_b.results)} expressions):")
    print(model_b.results[["Expression", "loss_train", "size"]].to_string(index=False))
    print()
    db_stats(egraph_db, "egraph.db (after A+B)")
    db_stats(fit_b_db, "fit_b.db")

    # ===================================================================
    # Step 3: Query top-N from both datasets via reggression
    # ===================================================================
    print("\n" + "=" * 64)
    print("Step 3: Query top-N from both fit databases via reggression")
    print("=" * 64)

    N = 50  # large N to increase intersection probability
    egg = Reggression(dataset=data_a_csv, loss="MSE")

    top_a = egg.dbTop(egraph_db, N, fitDb=fit_a_db)
    top_b = egg.dbTop(egraph_db, N, fitDb=fit_b_db)

    print(f"\n  Top {N} from dataset A:")
    print(top_a[["Id", "Expression", "Fitness", "Size"]].head(10).to_string(index=False))
    print(f"  ... ({len(top_a)} total)")

    print(f"\n  Top {N} from dataset B:")
    print(top_b[["Id", "Expression", "Fitness", "Size"]].head(10).to_string(index=False))
    print(f"  ... ({len(top_b)} total)")

    # ===================================================================
    # Step 4: Find intersection (e-classes in both top-N lists)
    # ===================================================================
    print("\n" + "=" * 64)
    print("Step 4: Intersection — e-classes in both top-N lists")
    print("=" * 64)

    ids_a = set(top_a["Id"].astype(int))
    ids_b = set(top_b["Id"].astype(int))
    common_ids = ids_a & ids_b

    print(f"\n  Dataset A top-{N}: {len(ids_a)} unique e-classes")
    print(f"  Dataset B top-{N}: {len(ids_b)} unique e-classes")
    print(f"  Intersection:      {len(common_ids)} e-classes in both")

    if common_ids:
        # Build lookup for both tables
        lookup_a = {int(row.Id): row for _, row in top_a.iterrows()}
        lookup_b = {int(row.Id): row for _, row in top_b.iterrows()}

        print(f"\n  Common e-classes (ranked by average fitness):")
        common_data = []
        for eid in common_ids:
            expr = lookup_a[eid]["Expression"]
            fit_a_val = lookup_a[eid]["Fitness"]
            fit_b_val = lookup_b[eid]["Fitness"]
            size = lookup_a[eid]["Size"]
            avg_fit = (fit_a_val + fit_b_val) / 2
            common_data.append((eid, expr, fit_a_val, fit_b_val, avg_fit, size))

        # Sort by average fitness (higher = better for negative-MSE)
        common_data.sort(key=lambda x: x[4], reverse=True)

        rows = []
        for eid, expr, fa, fb, avg, sz in common_data:
            rows.append({
                "Id": eid,
                "Expression": expr,
                "Fit_A": f"{fa:.4f}",
                "Fit_B": f"{fb:.4f}",
                "Avg_Fit": f"{avg:.4f}",
                "Size": sz,
            })
        common_df = pd.DataFrame(rows)
        print(common_df.to_string(index=False))
    else:
        print("\n  No common e-classes in top-N.")
        print("  This can happen when the datasets have very different structure.")
        print("  Try increasing N or using more generations.")

    # ===================================================================
    # Step 5: Show the union (all unique e-classes across both)
    # ===================================================================
    print("\n" + "=" * 64)
    print("Step 5: Union — all unique e-classes across both datasets")
    print("=" * 64)

    all_ids = ids_a | ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"\n  Total unique e-classes: {len(all_ids)}")
    print(f"  Only in A's top-{N}:  {len(only_a)}")
    print(f"  Only in B's top-{N}:  {len(only_b)}")
    print(f"  In both:              {len(common_ids)}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  Dataset A ({elapsed_a:.1f}s): {len(model_a.results)} Pareto expressions")
    print(f"  Dataset B ({elapsed_b:.1f}s): {len(model_b.results)} Pareto expressions")
    print(f"  Shared egraph: {db_stats.__name__} (see above)")
    print(f"  Intersection: {len(common_ids)} e-classes in both top-{N}")
    print()
    print("  The shared egraph lets eggp on dataset B benefit from expressions")
    print("  discovered on dataset A. The intersection reveals expressions that")
    print("  work well across different datasets — useful for model selection")
    print("  and understanding which structure is universal vs dataset-specific.")
    print("=" * 64)

    # Cleanup
    for f in (data_a_csv, data_b_csv, egraph_db, fit_a_db, fit_b_db):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
