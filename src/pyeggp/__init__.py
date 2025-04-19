import atexit
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, List
import string
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

VERSION: str = "1.0.4"


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

def make_function(expression, loss="MSE"):
    def func(x, t):
        y = eval(expression)
        if loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif loss == "Poisson":
            return np.exp(y)
        return y
    return func

class PyEGGP(BaseEstimator, RegressorMixin):
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

    dumpTo : str, default=""
        If not empty, it will save the final e-graph into the filename.

    loadFrom : str, default=""
        If not empty, it will load an e-graph and resume the search.
        The user must ensure that the loaded e-graph is from the same
        dataset and loss function.

    Examples
    --------
    >>> from pyeggp import PyEGGP
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = PyEGGP()
    >>> estimator.fit(X, y)
    >>>
    >>> estimator = PyEGGP(loss="Bernoulli")
    >>> estimator.fit(X, y)
    """
    def __init__(self, gen = 100, nPop = 100, maxSize = 15, nTournament = 3, pc = 0.9, pm = 0.3, nonterminals = "add,sub,mul,div", loss = "MSE", optIter = 50, optRepeat = 2, nParams = -1, split = 1, simplify = False, dumpTo = "", loadFrom = ""):
        nts = "add,sub,mul,div,power,powerabs,\
               aq,abs,sin,cos,tan,sinh,cosh,tanh,\
               asin,acos,atan,asinh,acosh,atanh,sqrt,\
               sqrtabs,cbrt,square,log,logabs,exp,recip,cube"
        losses = ["MSE", "Gaussian", "Bernoulli", "Poisson"]
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
        if split < 1:
            raise ValueError('split must be equal or greater than 1')
        if not isinstance(simplify, bool):
            raise TypeError('simplify must be a boolean')
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
        ''' Fits the regression model.

        Parameters
        ----------
        X : np.array
            An m x n np.array describing m observations of n features.
        y : np.array
            An np.array of size m with the measured target values.
        '''
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
        ''' Fits a multi-view regression model.

        Parameters
        ----------
        Xs : list(np.array)
            A list with k elements of m_k x n np.arrays describing m_k observations of n features.
        ys : list(np.array)
            A list of k elements of np.arrays of size m_k with the measured target values.
        '''
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
        return self.evaluate_best_model_view(X, view)

    def evaluate_best_model(self, x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        t = np.array(list(map(float, self.results.iloc[-1].theta.split(";"))))
        y = eval(self.results.iloc[-1].Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y
    def evaluate_best_model_view(self, x, view):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        ix = self.results.iloc[-1].id
        best = self.results[self.results.id==ix].iloc[view]
        t = np.array(list(map(float, best.theta.split(";"))))
        y = eval(best.Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y

    def evaluate_model_view(self, x, ix, view):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        best = self.results[self.results.id==ix].iloc[view]
        t = np.array(list(map(float, best.theta.split(";"))))
        y = eval(best.Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y
    def evaluate_model(self, ix, x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        t = np.array(list(map(float, self.results.iloc[-1].theta.split(";"))))
        y = eval(self.results.iloc[i].Numpy)
        if self.loss == "Bernoulli":
            return 1/(1 + np.exp(-y))
        elif self.loss == "Poisson":
            return np.exp(y)
        return y
    def score(self, X, y):
        ''' Calculates the score (single-view only).
        '''
        ypred = self.evaluate_best_model(X)
        return r2_score(y, ypred)
    def get_model(self, idx):
        ''' Get a `model` function and its visual representation. '''
        alphabet = list(string.ascii_uppercase)
        row = self.results[self.results['id']==idx].iloc[0]
        visual_expression = row['Numpy']
        model = make_function(visual_expression, self.loss)
        n_params_used = len(row['theta'].split(sep=';'))
    
        # Works for solutions with less than 26 parameters
        for i in range(n_params_used):
            visual_expression = visual_expression.replace(f't[{i}]', alphabet[i])
    
        # Works for data with less than 50 dimensions
        for i in range(50):
            visual_expression = visual_expression.replace(f'x[:, {i}]', f'X{i}')
    
        return model, visual_expression
