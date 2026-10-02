"""
db_ci_example.py - eggp DB mode with confidence intervals
=========================================================

This tutorial demonstrates the eggp DB-backed mode with profile-likelihood
confidence intervals in the Pareto front output.

The CI columns (ci_t0_lower, ci_t0_upper, ...) show the 95% profile-likelihood
interval for each parameter. Narrow intervals indicate well-estimated parameters;
wide intervals indicate high uncertainty.

Run from the ``tutorials/`` directory::

    python db_ci_example.py
"""

import os
import time
import sqlite3

import numpy as np
import pandas as pd

from eggp import EGGP


def make_dataset(path):
    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.uniform(-3, 3, n)
    x1 = rng.uniform(-3, 3, n)
    y  = np.sin(x0) + 0.5 * x1
    df = pd.DataFrame({"x0": x0, "x1": x1, "y": y})
    df.to_csv(path, index=False)
    return df


def main():
    data_csv = "db_ci_data.csv"
    db_file  = "db_ci.db"

    for f in (data_csv, db_file):
        if os.path.exists(f):
            os.remove(f)

    df = make_dataset(data_csv)
    X = df[["x0", "x1"]]
    y = df["y"]

    print("Running eggp with DB mode and CI computation...")
    print("The Pareto front CSV will include confidence interval columns.\n")

    model = EGGP(
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
        dbFile=db_file,
        dbDataset="ci_demo",
        dbFlushEvery=5,
    )

    t0 = time.time()
    model.fit(X, y)
    elapsed = time.time() - t0

    print(f"\nFinished in {elapsed:.1f}s")
    print(f"\nPareto front ({len(model.results)} expressions):")

    # Show the results with CI columns
    cols_to_show = ["Expression", "loss_train", "size"]
    ci_cols = [c for c in model.results.columns if c.startswith("ci_")]
    if ci_cols:
        cols_to_show.extend(ci_cols)
        print(model.results[cols_to_show].to_string(index=False))
        print(f"\nCI columns: {', '.join(ci_cols)}")
        print("Each parameter has lower/upper bounds showing the 95%")
        print("profile-likelihood confidence interval.")
    else:
        print(model.results[["Expression", "loss_train", "size"]].to_string(index=False))
        print("\n(No CI columns in output — check eggp version)")

    # Cleanup
    for f in (data_csv, db_file):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
