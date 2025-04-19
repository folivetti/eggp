# PyEGGP

A Python package for symbolic regression using e-graph-based genetic programming. PyEGGP provides a scikit-learn compatible API for evolutionary symbolic regression tasks.

More info [here](https://github.com/folivetti/srtree/tree/main/apps/eggp)

## Installation

```bash
pip install pyeggp
```

## Features

- Scikit-learn compatible API with `fit()` and `predict()` methods
- Genetic programming approach with e-graph representation
- Support for **multi-view symbolic regression** [see here](https://arxiv.org/abs/2402.04298)
- Customizable evolutionary parameters (population size, tournament selection, etc.)
- Flexible function set selection
- Various loss functions for different problem types
- Parameter optimization with multiple restarts
- Optional expression simplification through equality saturation
- Ability to save and load e-graphs

## Usage

### Basic Example

```python
from pyeggp import PyEGGP
import numpy as np

# Create sample data
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 3 * np.sin(X.ravel()) + np.random.normal(0, 1, 100)

# Create and fit the model
model = PyEGGP(gen=100, nonterminals="add,sub,mul,div,sin,cos")
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Examine the results
print(model.results)
```

### Multi-View Symbolic Regression

```python
from pyeggp import PyEGGP
import numpy as np

# Create multiple views of data
X1 = np.linspace(-5, 5, 50).reshape(-1, 1)
y1 = np.sin(X1.ravel()) + np.random.normal(0, 0.1, 50)

X2 = np.linspace(0, 10, 100).reshape(-1, 1)
y2 = np.sin(X2.ravel()) + np.random.normal(0, 0.2, 100)

# Create and fit multi-view model
model = PyEGGP(gen=150, nPop=200)
model.fit_mvsr([X1, X2], [y1, y2])

# Make predictions for each view
y_pred1 = model.predict_mvsr(X1, view=0)
y_pred2 = model.predict_mvsr(X2, view=1)
```

### Integration with scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pyeggp import PyEGGP

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and fit model
model = PyEGGP(gen=150, nPop=150, optIter=100)
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

## Available Functions

The following functions can be used in the `nonterminals` parameter:

- Basic operations: `add`, `sub`, `mul`, `div`
- Powers: `power`, `powerabs`, `square`, `cube`
- Roots: `sqrt`, `sqrtabs`, `cbrt`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Others: `abs`, `log`, `logabs`, `exp`, `recip`, `aq` (analytical quotient)

## Methods

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

## Results

After fitting, the `results` attribute contains a pandas DataFrame with details about the discovered models, including:
- Mathematical expressions
- Model complexity
- Parameter values
- Error metrics
- NumPy-compatible expressions

## License

[LICENSE]

## Citation

If you use PyEGGP in your research, please cite:

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

The bindings were created following the amazing example written by [wenkokke](https://github.com/wenkokke/example-haskell-wheel)
