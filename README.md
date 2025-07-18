# eggp - e-graph GP

*eggp* (**e**-**g**raph **g**enetic **p**rogramming), follows the same structure as the traditional GP. The initial population is created using ramped half-and-half respecting a maximum size and maximum depth parameter and, for a number of generations, it will choose two parents using tournament selection, apply the subtree crossover with probability $pc$ followed by the subtree mutation with probability $pm$, when the offsprings replace the current population following a dominance criteria.

The key differences of *eggp* are:

    - new solutions are inserted into the e-graph followed by one step of equality saturation to find and store some of the  equivalent expressions of the new offspring.
    - the current population is replaced by the set of individuals formed by: the Pareto front, the next front after excluding the first Pareto-front, and a selection of the last offspring at random until it reaches the desired population size.
    - the subtree crossover and mutation are modified to try to generate an unvisited exp
    
This repository provides a CLI and a Python package for eggp with a scikit-learn compatible API for symbolic regression.

Instructions:

- [CLI version](#cli)
- [Python version](#python)

## CLI

### How to use 

```bash
eggp - E-graph Genetic Programming for Symbolic Regression.

Usage: eggp (-d|--dataset INPUT-FILE) [-t|--test ARG] [-g|--generations GENS]
            (-s|--maxSize ARG) [-k|--split ARG] [--print-pareto] [--trace] 
            [--loss ARG] [--opt-iter ARG] [--opt-retries ARG] 
            [--number-params ARG] [--nPop ARG] [--tournament-size ARG] 
            [--pc ARG] [--pm ARG] [--non-terminals ARG] [--dump-to ARG] 
            [--load-from ARG] [--moo]

  An implementation of GP with modified crossover and mutation operators
  designed to exploit equality saturation and e-graphs.
  https://arxiv.org/abs/2501.17848

Available options:
  -d,--dataset INPUT-FILE  CSV dataset.
  -t,--test ARG            test data (default: "")
  -g,--generations GENS    Number of generations. (default: 100)
  -s,--maxSize ARG         max-size.
  -k,--split ARG           k-split ratio training-validation (default: 1)
  --print-pareto           print Pareto front instead of best found expression
  --trace                  print all evaluated expressions.
  --loss ARG               loss function: MSE, Gaussian, Poisson, Bernoulli.
                           (default: MSE)
  --opt-iter ARG           number of iterations in parameter optimization.
                           (default: 30)
  --opt-retries ARG        number of retries of parameter fitting. (default: 1)
  --number-params ARG      maximum number of parameters in the model. If this
                           argument is absent, the number is bounded by the
                           maximum size of the expression and there will be no
                           repeated parameter. (default: -1)
  --nPop ARG               population size (Default: 100). (default: 100)
  --tournament-size ARG    tournament size. (default: 2)
  --pc ARG                 probability of crossover. (default: 1.0)
  --pm ARG                 probability of mutation. (default: 0.3)
  --non-terminals ARG      set of non-terminals to use in the search.
                           (default: "Add,Sub,Mul,Div,PowerAbs,Recip")
  --dump-to ARG            dump final e-graph to a file. (default: "")
  --load-from ARG          load initial e-graph from a file. (default: "")
  --moo                    replace the current population with the pareto front
                           instead of replacing it with the generated children.
  -h,--help                Show this help text
```

The dataset file must contain a header with each features name, and the `--dataset` and `--test` arguments can be accompanied by arguments separated by ':' following the format:

`filename.ext:start_row:end_row:target:features`

where each ':' field is optional. The fields are:

- **start_row:end_row** is the range of the training rows (default 0:nrows-1).
   every other row not included in this range will be used as validation
- **target** is either the name of the  (if the datafile has headers) or the index
   of the target variable
- **features** is a comma separated list of names or indices to be used as
  input variables of the regression model.

Example of valid names: `dataset.csv`, `mydata.tsv`, `dataset.csv:20:100`, `dataset.tsv:20:100:price:m2,rooms,neighborhood`, `dataset.csv:::5:0,1,2`.

The format of the file will be determined by the extension (e.g., csv, tsv,...). To use multi-view, simply pass multiple filenames in double-quotes:

```bash
eggp --dataset "dataset1.csv dataset2.csv dataset3.csv" ...
```

### Installation 

To install eggp you'll need:

- `libz`
- `libnlopt`
- `libgmp`
- `ghc-9.6.6`
- `cabal` or `stack`

### Method 1: PIP

Simply run:

```bash
pip install eggp 
```

under your Python environment.

### Method 2: cabal

After installing the dependencies (e.g., `apt install libz libnlopt libgmp`), install [`ghcup`](https://www.haskell.org/ghcup/#)

For Linux, macOS, FreeBSD or WSL2:

```bash 
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

For Windows, run the following in a PowerShell:

```bash
Set-ExecutionPolicy Bypass -Scope Process -Force;[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; try { & ([ScriptBlock]::Create((Invoke-WebRequest https://www.haskell.org/ghcup/sh/bootstrap-haskell.ps1 -UseBasicParsing))) -Interactive -DisableCurl } catch { Write-Error $_ }
```

After the installation, run `ghcup tui` and install the latest `stack` or `cabal` together with `ghc-9.6.6` (select the items and press `i`).
To install `srsimplify` simply run:

```bash 
cabal install
```

## Python

### Features

- Scikit-learn compatible API with `fit()` and `predict()` methods
- Genetic programming approach with e-graph representation
- Support for **multi-view symbolic regression** [see here](https://arxiv.org/abs/2402.04298)
- Customizable evolutionary parameters (population size, tournament selection, etc.)
- Flexible function set selection
- Various loss functions for different problem types
- Parameter optimization with multiple restarts
- Optional expression simplification through equality saturation
- Ability to save and load e-graphs

### Usage

### Basic Example

```python
from eggp import EGGP
import numpy as np

# Create sample data
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 3 * np.sin(X.ravel()) + np.random.normal(0, 1, 100)

# Create and fit the model
model = EGGP(gen=100, nonterminals="add,sub,mul,div,sin,cos")
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Examine the results
print(model.results)
```

### Multi-View Symbolic Regression

```python
from eggp import EGGP
import numpy as np

# Create multiple views of data
X1 = np.linspace(-5, 5, 50).reshape(-1, 1)
y1 = np.sin(X1.ravel()) + np.random.normal(0, 0.1, 50)

X2 = np.linspace(0, 10, 100).reshape(-1, 1)
y2 = np.sin(X2.ravel()) + np.random.normal(0, 0.2, 100)

# Create and fit multi-view model
model = EGGP(gen=150, nPop=200)
model.fit_mvsr([X1, X2], [y1, y2])

# Make predictions for each view
y_pred1 = model.predict_mvsr(X1, view=0)
y_pred2 = model.predict_mvsr(X2, view=1)
```

### Integration with scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from eggp import EGGP

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and fit model
model = EGGP(gen=150, nPop=150, optIter=100)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gen` | int | 100 | Number of generations to run |
| `nPop` | int | 100 | Population size |
| `maxSize` | int | 15 | Maximum allowed size for expressions (max 100) |
| `nTournament` | int | 3 | Tournament size for parent selection |
| `pc` | float | 0.9 | Probability of performing crossover |
| `pm` | float | 0.3 | Probability of performing mutation |
| `nonterminals` | str | "add,sub,mul,div" | Comma-separated list of allowed functions |
| `loss` | str | "MSE" | Loss function: "MSE", "Gaussian", "Bernoulli", or "Poisson" |
| `optIter` | int | 50 | Number of iterations for parameter optimization |
| `optRepeat` | int | 2 | Number of restarts for parameter optimization |
| `nParams` | int | -1 | Maximum number of parameters (-1 for unlimited) |
| `split` | int | 1 | Data splitting ratio for validation |
| `simplify` | bool | False | Whether to apply equality saturation to simplify expressions |
| `dumpTo` | str | "" | Filename to save the final e-graph |
| `loadFrom` | str | "" | Filename to load an e-graph to resume search |

### Available Functions

The following functions can be used in the `nonterminals` parameter:

- Basic operations: `add`, `sub`, `mul`, `div`
- Powers: `power`, `powerabs`, `square`, `cube`
- Roots: `sqrt`, `sqrtabs`, `cbrt`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Others: `abs`, `log`, `logabs`, `exp`, `recip`, `aq` (analytical quotient)

### Methods

### Core Methods
- `fit(X, y)`: Fits the symbolic regression model
- `predict(X)`: Generates predictions using the best model
- `score(X, y)`: Computes R² score of the best model

### Multi-View Methods
- `fit_mvsr(Xs, ys)`: Fits a multi-view regression model
- `predict_mvsr(X, view)`: Generates predictions for a specific view
- `evaluate_best_model_view(X, view)`: Evaluates the best model on a specific view
- `evaluate_model_view(X, ix, view)`: Evaluates a specific model on a specific view

### Utility Methods
- `evaluate_best_model(X)`: Evaluates the best model on the given data
- `evaluate_model(ix, X)`: Evaluates the model with index `ix` on the given data
- `get_model(idx)`: Returns a model function and its visual representation

### Results

After fitting, the `results` attribute contains a pandas DataFrame with details about the discovered models, including:
- Mathematical expressions
- Model complexity
- Parameter values
- Error metrics
- NumPy-compatible expressions

## License

[LICENSE]

## Citation

If you use EGGP in your research, please cite:

```
@inproceedings{eggp,
author = {de Franca, Fabricio Olivetti and Kronberger, Gabriel},
title = {Improving Genetic Programming for Symbolic Regression with Equality Graphs},
year = {2025},
isbn = {9798400714658},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712256.3726383},
doi = {10.1145/3712256.3726383},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
pages = {},
numpages = {9},
keywords = {Symbolic regression, Genetic programming, Equality saturation, Equality graphs},
location = {Malaga, Spain},
series = {GECCO '25},
archivePrefix = {arXiv},
       eprint = {2501.17848},
 primaryClass = {cs.LG}, 
}
```

## Acknowledgments

The bindings were created following the amazing example written by [wenkokke](https://github.com/wenkokke/example-haskell-wheel)

Fabricio Olivetti de Franca is supported by Conselho Nacional de Desenvolvimento Cient\'{i}fico e Tecnol\'{o}gico (CNPq) grant 301596/2022-0.

Gabriel Kronberger is supported by the Austrian Federal Ministry for Climate Action, Environment, Energy, Mobility, Innovation and Technology, the Federal Ministry for Labour and Economy, and the regional government of Upper Austria within the COMET project ProMetHeus (904919) supported by the Austrian Research Promotion Agency (FFG). 
