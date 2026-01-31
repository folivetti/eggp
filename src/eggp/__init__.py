import atexit
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, List
import string
from io import StringIO
import tempfile
import csv
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from ._binding import (
    unsafe_hs_eggp_version,
    unsafe_hs_eggp_main,
    unsafe_hs_eggp_run,
    unsafe_hs_eggp_init,
    unsafe_hs_eggp_exit,
)

VERSION: str = "1.0.12"


_hs_rts_init: bool = False
_hs_rts_lock: Lock = Lock()


def hs_rts_exit() -> None:
    global _hs_rts_lock
    with _hs_rts_lock:
        unsafe_hs_eggp_exit()


@contextmanager
def hs_rts_init(args: List[str] = []) -> Iterator[None]:
    global _hs_rts_init
    global _hs_rts_lock
    with _hs_rts_lock:
        if not _hs_rts_init:
            _hs_rts_init = True
            unsafe_hs_eggp_init(args)
            atexit.register(hs_rts_exit)
    yield None


def version() -> str:
    with hs_rts_init():
        return unsafe_hs_eggp_version()


def main(args: List[str] = []) -> int:
    with hs_rts_init(args):
        return unsafe_hs_eggp_main()

def eggp_run(dataset: str, gen: int, nPop: int, maxSize: int, nTournament: int, pc: float, pm: float, nonterminals: str, loss: str, optIter: int, optRepeat: int, nParams: int, split: int, max_time : int, simplify: int, trace : int, generational : int, dumpTo: str, loadFrom: str, varnames : str) -> str:
    with hs_rts_init():
        return unsafe_hs_eggp_run(dataset, gen, nPop, maxSize, nTournament, pc, pm, nonterminals, loss, optIter, optRepeat, nParams, split, max_time, simplify, trace, generational, dumpTo, loadFrom, varnames)

def make_function(expression, loss="MSE"):
    def func(x, t):
        y = eval(expression)
        if loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif loss == "Poisson":
            return np.exp(y)
        return y
    return func

class EGGP(BaseEstimator, RegressorMixin):
    """ Builds a symbolic regression model using eggp.

    Parameters
    ----------
    gen : int, default=100
        The number of generations.

    nPop : int, default=100
        Population size.

    maxSize : int, default=15
        Maximum allowed size for the expression.
        This should not be larger than 100 as the e-graph may grow
        too large.

    nTournament : int, default=3
        Tournament size. During parent selection it will
        pick `nTournament` expressions at random and
        return the best among them.

    pc : float, default=0.9
        Probability of performing the crossover operator.

    pm : float, default=0.3
        Probability of performing the mutation operator.

    nonterminals : str, default="add,sub,mul,div"
        String of a comma separated list of nonterminals.
        These are the allowed functions to be used during the search.
        Available functions: add,sub,mul,div,power,powerabs,aq,abs,sin,cos,
                             tan,sinh,cosh,tanh,asin,acos,atan,asinh,acosh,
                             atanh,sqrt,sqrtabs,cbrt,square,log,logabs,exp,
                             recip,cube.
        Where `aq` is the analytical quotient (x/sqrt(1 + y^2)),
              `powerabs` is the protected power (x^|y|)
              `sqrtabs` is the protected sqrt (sqrt(|x|))
              `logabs` is the protected log (log(|x|))
              `recip` is the reciprocal (1/x)
              `cbrt` is the cubic root

    loss : {"MSE", "Gaussian", "Bernoulli", "Poisson"}, default="MSE"
        Loss function used to evaluate the expressions:
        - MSE (mean squared error) should be used for regression problems.
        - Gaussian likelihood should be used for regression problem when you want to
          fit the error term.
        - Bernoulli likelihood should be used for classification problem.
        - Poisson likelihood should be used when the data distribution follows a Poisson.

    optIter : int, default=50
        Number of iterations for the parameter optimization.

    optRepeat : int, default=2
        Number of restarts for the parameter optimization.

    nParams : int, default=-1
        Maximum number of parameters. If set to -1 it will
        allow the expression to have any number of parameters.
        If set to a number > 0, it will limit the number of parameters,
        but allow it to appear multiple times in the expression.
        E.g., t0 * x0 + exp(t0*x0 + t1)

    split : int, default=1
        How to split the data to create the validation set.
        If set to 1, it will use the whole data for fitting the parameter and
        calculating the fitness function.
        If set to n>1, it will use 1/n for calculating the fitness function
        and the reminder for fitting the parameter.

    simplify : bool, default=False
        Whether to apply a final step of equality saturation to simplify the expressions.

    trace : bool, default=False
        Whether to return a pandas dataframe of all visited expressions.

    generational : bool, default=False
        Whether to use generational (replacing the currente generation with the new one) instead
        of keeping the Pareto front.

    dumpTo : str, default=""
        If not empty, it will save the final e-graph into the filename.

    loadFrom : str, default=""
        If not empty, it will load an e-graph and resume the search.
        The user must ensure that the loaded e-graph is from the same
        dataset and loss function.

    Examples
    --------
    >>> from eggp import EGGP
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = EGGP()
    >>> estimator.fit(X, y)
    >>>
    >>> estimator = EGGP(loss="Bernoulli")
    >>> estimator.fit(X, y)
    """
    def __init__(self, gen = 100, nPop = 100, maxSize = 15, nTournament = 3, pc = 0.9, pm = 0.3, nonterminals = "add,sub,mul,div", loss = "MSE", optIter = 50, optRepeat = 2, nParams = -1, folds = 1, max_time = -1, simplify = False, trace = False, generational = False, dumpTo = "", loadFrom = ""):
        nts = "add,sub,mul,div,power,powerabs,\
               aq,abs,sin,cos,tan,sinh,cosh,tanh,\
               asin,acos,atan,asinh,acosh,atanh,sqrt,\
               sqrtabs,cbrt,square,log,logabs,exp,recip,cube"
        losses = ["MSE", "LOG10", "Gaussian", "Bernoulli", "Poisson"]
        if gen < 1:
            raise ValueError('gen should be greater than 1')
        if nPop < 1:
            raise ValueError('nPop should be greater than 1')
        if maxSize < 1 or maxSize > 100:
            raise ValueError('maxSize should be a value between 1 and 100')
        if nTournament < 1 or nTournament > nPop:
            raise ValueError('nTournament should be a value between 1 and nPop')
        if pc < 0 or pc > 1:
            raise ValueError('pc should be between 0 and 1')
        if pm < 0 or pm > 1:
            raise ValueError('pm should be between 0 and 1')
        if any(t not in nts for t in nonterminals):
            raise ValueError('nonterminals must be a comma separated list of one or more of ', nts)
        if loss not in losses:
            raise ValueError('loss must be one of ', losses)
        if optIter < 0:
            raise ValueError('optIter must be a positive number')
        if optRepeat < 0:
            raise ValueError('optRepeat must be a positive number')
        if nParams < -1:
            raise ValueError('nParams must be either -1 or a positive number')
        if folds < 1:
            raise ValueError('folds must be equal or greater than 1')
        if not isinstance(simplify, bool):
            raise TypeError('simplify must be a boolean')
        if not isinstance(trace, bool):
            raise TypeError('trace must be a boolean')
        if not isinstance(generational, bool):
            raise TypeError('generational must be a boolean')
        if not isinstance(max_time, int):
            raise TypeError('max_time must be an integer')
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
        self.folds = folds
        self.max_time = max_time
        self.simplify = int(simplify)
        self.trace = int(trace)
        self.generational = int(generational)
        self.dumpTo = dumpTo
        self.loadFrom = loadFrom
        self.is_fitted_ = False

    def combine_dataset(self, X, y, Xerr, yerr):
        ''' Combines the error information into a single dataset.

        Parameters
        ----------
        X : np.array
            An m x n np.array describing m observations of n features.
        y : np.array
            An np.array of size m with the measured target values.
        Xerr : np.array
               An m x n np.array with the measurement errors of the features.
        yerr : np.array
               An np.array of size m with the measurement errors of the target.

        Returns the combined dataset
        '''
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        if isinstance(Xerr, pd.DataFrame):
            Xerr = Xerr.to_numpy()
        if isinstance(yerr, pd.DataFrame):
            yerr = yerr.to_numpy()

        if X.ndim == 1:
            X = X.reshape(-1,1)
        y = y.reshape(-1, 1)
        stacked = [X, y]
        if Xerr is not None and self.loss == "ROXY":
            Xerr = Xerr.reshape(-1, 1)
            stacked.append(Xerr)
        if yerr is not None and self.loss in ["ROXY", "HGaussian"]:
            yerr = yerr.reshape(-1, 1)
            stacked.append(yerr)
        if self.loss == "ROXY":
            stacked += [np.log10(X).reshape(-1,1), np.log10(y).reshape(-1,1), np.square(Xerr/(Xerr * np.log(10))).reshape(-1,1), np.square(yerr/(yerr*np.log(10))).reshape(-1,1)]
        return np.hstack(stacked)

    def get_header(self, ndim):
        if self.loss != "ROXY":
            return [f"x{i}" for i in range(ndim)] + ["y"]
        elif self.loss == "HGaussian":
            return [f"x{i}" for i in range(ndim)] + ["y", "yerr"]
        else:
            return ["gbar","gobs","e_gbar","e_gobs","logX","logY","logXErr","logYErr"]

    def get_fname(self, dname, header):
        if self.loss == "ROXY":
            return f"{dname}:::gobs:gbar,logX,logY,logXErr,logYErr:e_gobs:e_gbar"
        elif self.loss == "HGaussian":
            varnames = ",".join(header[:-2])
            return f"{dname}:::y:{varnames}:yerr"
        else:
            return dname

    def fit(self, X, y, Xerr = None, yerr = None):
        ''' Fits the regression model.

        Parameters
        ----------
        X : np.array
            An m x n np.array describing m observations of n features.
        y : np.array
            An np.array of size m with the measured target values.
        Xerr : np.array
               An m x n np.array with the measurement errors of the features.
        yerr : np.array
               An np.array of size m with the measurement errors of the target.

        A table with the fitted models and additional information
        will be stored as a Pandas dataframe in self.results.
        '''
        combined = self.combine_dataset(X, y, Xerr, yerr)
        header = self.get_header(X.shape[1])
        if isinstance(X, pd.DataFrame):
            varnames = ",".join(X.columns)
        else:
            varnames = ""

        with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False, prefix='datatemp_', suffix='.csv', dir=os.getcwd()) as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow(header)
            writer.writerows(combined)
            dataset = temp_file.name
        dname = self.get_fname(dataset, header)

        try:
            csv_data = eggp_run(dname, self.gen, self.nPop, self.maxSize, self.nTournament, self.pc, self.pm, self.nonterminals, self.loss, self.optIter, self.optRepeat, self.nParams, self.folds, self.max_time, self.simplify, self.trace, self.generational, self.dumpTo, self.loadFrom, varnames)

        finally:
            os.remove(dataset)

        if len(csv_data) > 0:
            csv_io = StringIO(csv_data.strip())
            self.results = pd.read_csv(csv_io, header=0, dtype={'theta':str}, keep_default_na=False)
            self.is_fitted_ = True
        return self

    def fit_mvsr(self, Xs, ys, Xerrs = None, yerrs = None):
        ''' Fits a multi-view regression model.

        Parameters
        ----------
        Xs : list(np.array)
            A list with k elements of m_k x n np.arrays describing m_k observations of n features.
        ys : list(np.array)
            A list of k elements of np.arrays of size m_k with the measured target values.
        '''
        if Xerrs is None:
            Xerrs = [None for _ in Xs]
        if yerrs is None:
            yerrs = [None for _ in ys]

        combineds = [self.combine_dataset(X, y, Xerr, yerr) for X, y, Xerr, yerr in zip(Xs, ys, Xerrs, yerrs)]
        header = self.get_header(Xs[0].shape[1])
        datasets = []
        datasetsNames = []
        if isinstance(Xs[0], pd.DataFrame):
            varnames = ",".join(Xs[0].columns)
        else:
            varnames = ""

        for combined in combineds:
            with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False, prefix='datatemp_', suffix='.csv', dir=os.getcwd()) as temp_file:
                writer = csv.writer(temp_file)
                writer.writerow(header)
                writer.writerows(combined)
                datasetsNames.append(temp_file.name)
                datasets.append(self.get_fname(temp_file.name, header))

        try:
            csv_data = eggp_run(" ".join(datasets), self.gen, self.nPop, self.maxSize, self.nTournament, self.pc, self.pm,
                                self.nonterminals, self.loss, self.optIter, self.optRepeat, self.nParams, self.folds, self.max_time, self.simplify, self.trace, self.generational, self.dumpTo, self.loadFrom, varnames)
        finally:
            for dataset in datasetsNames:
                os.remove(dataset)

        if len(csv_data) > 0:
            csv_io = StringIO(csv_data.strip())
            self.results = pd.read_csv(csv_io, header=0, dtype={'theta':str}, keep_default_na=False)
            self.is_fitted_ = True
        return self

    def predict(self, X):
        ''' Generates the prediction using the best model (selected by accuracy)

        Parameters
        ----------
        X : np.array
            An m x n np.array describing m observations of n features.
            This array must have the same number of features as the training data.

        Return
        ------
        y : np.array
            A vector of predictions

        A table with the fitted models and additional information
        will be stored as a Pandas dataframe in self.results.
        '''
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return self.evaluate_best_model(X)

    def predict_mvsr(self, X, view):
        ''' Generates the prediction using the best model (selected by accuracy)
            of the sepecified `view`

        Parameters
        ----------
        X : np.array
            An m x n np.array describing m observations of n features.
            This array must have the same number of features as the training data.

        view : int
            The index of the view (starting at 0).

        Return
        ------
        y : np.array
            A vector of predictions
        '''
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return self.evaluate_best_model_view(X, view)

    def evaluate_best_model(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if x.ndim == 1:
            x = x.reshape(-1,1)
        tStr = self.results.iloc[-1].theta.split(";")
        t = np.array(list(map(float, tStr))) if len(tStr[0]) > 0  else np.array([])
        y = eval(self.results.iloc[-1].Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y
    def evaluate_best_model_view(self, x, view):
        if view not in np.unique(self.results.view.values):
            raise ValueError("Invalid view index")
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if x.ndim == 1:
            x = x.reshape(-1,1)
        ix = self.results.iloc[-1].id
        best = self.results[self.results.id==ix].iloc[view]
        t = np.array(list(map(float, best.theta.split(";")))) if len(best.theta) > 0 else np.array([])
        y = eval(best.Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y

    def evaluate_model_view(self, x, ix, view):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if ix not in np.unique(self.results.id.values):
            raise ValueError("Invalid model index")
        if view not in np.unique(self.results.view.values):
            raise ValueError("Invalid view index")

        best = self.results[self.results.id==ix].iloc[view]
        t = np.array(list(map(float, best.theta.split(";")))) if len(best.theta) > 0 else np.array([])
        y = eval(best.Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y
    def evaluate_model(self, ix, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if ix not in np.unique(self.results.id.values):
            raise ValueError("Invalid model index")

        tStr = self.results.iloc[ix].theta.split(";")
        t = np.array(list(map(float, tStr))) if len(tStr[0]) > 0 else np.array([])
        y = eval(self.results.iloc[ix].Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y
    def score(self, X, y):
        ''' Calculates the score (single-view only).
        '''
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        ypred = self.evaluate_best_model(X)
        return r2_score(y, ypred)
    def get_model(self, idx):
        ''' Get a `model` function and its visual representation. From srbench. '''
        if idx not in np.unique(self.results.id.values):
            raise ValueError("Invalid model index")

        alphabet = list(string.ascii_uppercase)
        row = self.results[self.results['id']==idx].iloc[0]
        visual_expression = row['Numpy']
        model = make_function(visual_expression, self.loss)
        n_params_used = len(row['theta'].split(sep=';')) if len(row['theta']) > 0 else 0
    
        # Works for solutions with less than 26 parameters
        for i in range(n_params_used):
            visual_expression = visual_expression.replace(f't[{i}]', alphabet[i])
    
        # Works for data with less than 50 dimensions
        for i in range(50):
            visual_expression = visual_expression.replace(f'x[:, {i}]', f'X{i}')
    
        return model, visual_expression
    def cal_plot(self, X, y, ix):
        if ix not in np.unique(self.results.id.values):
            raise ValueError("Invalid model index")

        yhat = self.evaluate_model(ix, X)
        y_lo = y.min() - 0.1*(y.max() - y.min())
        y_hi = y.max() + 0.1*(y.max() - y.min())
        rng  = np.arange(y_lo, y_hi, 0.01)
        plt.plot(rng, rng)
        plt.plot(yhat, y, '.')
        plt.xlabel(r"$\hat{y}$")
        plt.ylabel(r"$y$")
        plt.ylim(y_lo, y_hi)
        plt.xlim(y_lo, y_hi)
