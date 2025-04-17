import atexit
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, List
from io import StringIO
import tempfile
import csv

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error, r2_score

from ._binding import (
    unsafe_hs_pyeggp_version,
    unsafe_hs_pyeggp_main,
    unsafe_hs_pyeggp_run,
    unsafe_hs_pyeggp_init,
    unsafe_hs_pyeggp_exit,
)

VERSION: str = "1.3.0"


_hs_rts_init: bool = False
_hs_rts_lock: Lock = Lock()


def hs_rts_exit() -> None:
    global _hs_rts_lock
    with _hs_rts_lock:
        unsafe_hs_pyeggp_exit()


@contextmanager
def hs_rts_init(args: List[str] = []) -> Iterator[None]:
    global _hs_rts_init
    global _hs_rts_lock
    with _hs_rts_lock:
        if not _hs_rts_init:
            _hs_rts_init = True
            unsafe_hs_pyeggp_init(args)
            atexit.register(hs_rts_exit)
    yield None


def version() -> str:
    with hs_rts_init():
        return unsafe_hs_pyeggp_version()


def main(args: List[str] = []) -> int:
    with hs_rts_init(args):
        return unsafe_hs_pyeggp_main()

def pyeggp_run(dataset: str, gen: int, nPop: int, maxSize: int, nTournament: int, pc: float, pm: float, nonterminals: str, loss: str, optIter: int, optRepeat: int, nParams: int, split: int, simplify: int, dumpTo: str, loadFrom: str) -> str:
    with hs_rts_init():
        return unsafe_hs_pyeggp_run(dataset, gen, nPop, maxSize, nTournament, pc, pm, nonterminals, loss, optIter, optRepeat, nParams, split, simplify, dumpTo, loadFrom)

class PyEGGP(BaseEstimator, RegressorMixin):
    def __init__(self, gen = 100, nPop = 100, maxSize = 15, nTournament = 3, pc = 0.9, pm = 0.3, nonterminals = "add,sub,mul,div", loss = "MSE", optIter = 50, optRepeat = 2, nParams = -1, split = 1, simplify = False, dumpTo = "", loadFrom = ""):
        self.gen = gen
        self.nPop = nPop
        self.maxSize = maxSize
        self.nTournament = nTournament
        self.pc = pc
        self.pm = pm
        self.nonterminals = nonterminals
        self.loss = loss
        self.optIter = optIter
        self.optRepeat = optRepeat
        self.nParams = nParams
        self.split = split
        self.simplify = int(simplify)
        self.dumpTo = dumpTo
        self.loadFrom = loadFrom
        self.is_fitted_ = False

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        y = y.reshape(-1, 1)
        combined = np.hstack([X, y])
        header = [f"x{i}" for i in range(X.shape[1])] + ["y"]
        with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False, suffix='.csv') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow(header)
            writer.writerows(combined)
            dataset = temp_file.name

        csv_data = pyeggp_run(dataset, self.gen, self.nPop, self.maxSize, self.nTournament, self.pc, self.pm, self.nonterminals, self.loss, self.optIter, self.optRepeat, self.nParams, self.split, self.simplify, self.dumpTo, self.loadFrom)
        if len(csv_data) > 0:
            csv_io = StringIO(csv_data.strip())
            self.results = pd.read_csv(csv_io, header=0)
            self.is_fitted_ = True
        return self

    def fit_mvsr(self, Xs, ys):
        if Xs[0].ndim == 1:
            Xs = [X.reshape(-1,1) for X in Xs]
        ys = [y.reshape(-1, 1) for y in ys]
        combineds = [np.hstack([X, y]) for X, y in zip(Xs, ys)]
        header = [f"x{i}" for i in range(Xs[0].shape[1])] + ["y"]
        datasets = []
        for combined in combineds:
            with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False, suffix='.csv') as temp_file:
                writer = csv.writer(temp_file)
                writer.writerow(header)
                writer.writerows(combined)
                datasets.append(temp_file.name)

        csv_data = pyeggp_run(" ".join(datasets), self.gen, self.nPop, self.maxSize, self.nTournament, self.pc, self.pm, self.nonterminals, self.loss, self.optIter, self.optRepeat, self.nParams, self.split, self.simplify, self.dumpTo, self.loadFrom)
        if len(csv_data) > 0:
            csv_io = StringIO(csv_data.strip())
            self.results = pd.read_csv(csv_io, header=0, dtype={'theta':str})
            self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.evaluate_best_model(X)

    def predict_mvsr(self, X, view):
        check_is_fitted(self)
        return self.evaluate_best_model_view(X, view)

    def evaluate_best_model(self, x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        t = np.array(list(map(float, self.results.iloc[-1].theta.split(";"))))
        return eval(self.results.iloc[-1].Numpy)
    def evaluate_best_model_view(self, x, view):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        ix = self.results.iloc[-1].id
        best = self.results[self.results.id==ix].iloc[view]
        t = np.array(list(map(float, best.theta.split(";"))))
        return eval(best.Numpy)
    def evaluate_model_view(self, x, ix, view):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        best = self.results[self.results.id==ix].iloc[view]
        t = np.array(list(map(float, best.theta.split(";"))))
        return eval(best.Numpy)
    def evaluate_model(self, ix, x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        t = np.array(list(map(float, self.results.iloc[-1].theta.split(";"))))
        return eval(self.results.iloc[i].Numpy)
    def score(self, X, y):
        ypred = self.evaluate_best_model(X)
        return r2_score(y, ypred)
