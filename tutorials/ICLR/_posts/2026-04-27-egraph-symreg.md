---
layout: distill
title: The Secret Weapon for Better Equation Discovery: e-graphs and Equality Saturation
description: How to use e-graphs and equality saturation to improve the exploration of symbolic regression search space.
date: 2026-04-27
future: true
htmlwidgets: true

# anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-egraph-symreg.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: [Introduction]
  - name: [It's all the same, no matter where you are]
  - name: [A Database for Math Expressions: Enters the e-graph]
  - name: [Equality Saturation: Automatically Generating Equivalence]
  - name: [The GPS for Symbolic Regression: e-graphs!]
  - name: [Generating uniqueness]
  - name: [Explore the Search Space with `eggp` and `rEGGression`]
  - name: [A Powerful Database for Equations: Using e-graphs and Equality Saturation for Interactive Equation Discovery]
  - name: [Laying the egg 🥚]
  - name: [Hatching the egg 🐣]
  - name: [With a little help from my friends 🐣🐤]
  - name: [It's all the same, no matter where you are 🐥🐥🐥]
---

## Introduction

In data science, physics, and engineering, the ultimate goal isn't just prediction—it's **understanding**. Finding a single, elegant mathematical formula that perfectly describes a set of data points is the holy grail. This is called **Equation Discovery** or, also, **Symbolic Regression (SR)**.

Traditional AI models like Neural Networks give us complex, black-box equations. SR aims for human-readable formulas (like $f(x) = \log(x) + c$).

To find this perfect formula, all search algorithms—whether they use Genetic Programming (GP), Monte Carlo Tree Search (MCTS), or Deep Learning (DL) —follow a a cycle of: **proposing** a candidate equation, **learning** from its performance, and repeat.

How the proposal and learning steps work depends on the algorithm:
- **Genetic Programming** proposes new equations by modifying existing equations or combining them. It learns by favoring the selection of the best equations found so far.
- **Monte Carlo Tree Search** proposal step generates a new equation by traversing a tree of possible grammar derivations that are more probable to fit the data taking a confidence interval into consideration. It learns by updating the probabilities of each derivation.
- **Deep Learning** and **Reinforcement Learning** proposes new equations by choosing the next symbol that maximizes the expected reward given the last choice. It learns by reinforcing the quality of the generated expression through the sequence of steps.

The problem? The search space is unbelievably vast and filled with redundancy.

## It's all the same, no matter where you are

Imagine trying to navigate a forest with many paths leading to the same (wrong) destination, and you have to try them all until you follow one that leads you to your goal. That's the reality of Symbolic Regression.
Consider the simple expression $2x$. How many different ways can you write that same value?

$$
x+x \\
\frac{4x}{2} \\
3x-x \\ 
\dots \text{and many more!}
$$

All these expressions are mathematically identical; they will all yield the exact same result for the same dataset. 

This redundancy creates two issues for the search algorithms:

1.  **Wasted Time:** The algorithm might revisit $x+x$ after having already explored $2x$, wasting valuable computational budget.
    
2.  **Complexity:** If $x+x$ is the correct solution, we want its simplest, and most interpretable form ($2x$), not one of the infinitely complex equivalent forms. Using post-processing simplification tools often fails or introduces new problems, as shown in <d-cite bibtex-key="de2023reducing"></d-cite>.
 
On the other hand, redundancy can be helpful. Sometimes, navigating from $x+x$ to $3x-x$ can be a "stepping stone" to reach a new, better area of the search space. This is known as the **neutral space theory**<d-cite bibtex-key="banzhaf2024combinatorics"></d-cite>.

But what if we could detect _all_ equivalent expressions in real-time and use that knowledge to make the search efficient?

## A Database for Math Expressions: Enters the e-graph

The solution lies in **e-graphs** and **Equality Saturation** <d-cite bibtex-key="tate2009equality"></d-cite>.

Think of an e-graph as a **smart database system** for mathematical expressions. It’s designed to store many different, but equivalent, expressions with a minimum amount of space. It also makes it easier to query for expressions with certain patterns.

In an e-graph, symbols (like $+, -, x, \log$) are called **e-nodes**. The core concept is the **e-class**, which acts as an **Equivalence Group**. Any e-node belonging to the same e-class represents a mathematically identical value.

For example, in the figure below, the dashed box in the middle is an e-class. It contains two e-nodes: one for multiplication (`2x`) and one for addition (`x+x`). Because they are in the same e-class, the graph automatically knows that $2x = x+x$.

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/blog1.png" class="img-fluid" %}

This structure is immensely powerful. Now, when the graph builds a larger expression, like a term squared (the very top multiplication operator in this e-graph), it knows it can be represented in four different ways instantly:

$$
(2x) (2x) \\ 
(2x) (x+x) \\ 
(x+x) (2x) \\ 
(x+x)(x+x) 
$$

The E-graph stores all four, but only pays the storage cost for one!

## Equality Saturation: Automatically Generating Equivalence 

How does the E-graph learn what's equivalent? It uses an algorithm called **Equality Saturation**. This process takes a simple set of mathematical rules (like the distributive property or $a+a=2a$) and applies them automatically until no new equivalences can be found (or until a time limit is reached).

Let’s watch it work on the expression $(x+x)^2$ using three simple rules:

$$
\alpha + \alpha \rightarrow 2\alpha \\
\alpha \times \alpha \rightarrow \alpha^2 \\
\alpha \times (\beta + \gamma) \rightarrow (\alpha \times \beta + \alpha \times \gamma)
$$

1. **Start:** Insert $(x+x)^2$ into the graph.
 {% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/blog2.png" class="img-fluid" %}

2. **Apply Rules:** The rule $\alpha + \alpha \rightarrow 2\alpha$ applies to the inner expression $x+x$:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/blog3.png" class="img-fluid" %}

3. We **insert** the right-hand side, $2x$ and **merge** with the e-class for $x+x$, as the graph knows they are identical:
{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/blog4.png" class="img-fluid" %}

4. **Repeat until Saturation:** The process continues, applying other rules until the E-graph contains every possible equivalent expression derived from these rules:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/blog5.png" class="img-fluid" %}

The most popular implementation of equality saturation is Egg<d-cite bibtex-key="willsey2021egg"></d-cite>, a library written in Rust.
But, how can e-graphs and equality saturation help with symbolic regression search??

## The GPS for Symbolic Regression: e-graphs!

A few years ago, some authors realized this powerful mechanism could be the missing piece in Symbolic Regression and  **pioneered** their integration in different situations.

First, they demonstrated that e-graphs are a superior **simplification tool** compared to standard methods like `sympy` <d-cite bibtex-key="de2023reducing"></d-cite>. By simplifying equations with equality saturation, we not only reduced model complexity but also increased the probability of finding the best-fitting local optima <d-cite bibtex-key="kronberger2024effects"></d-cite>.

Second, the e-graph structure could be used to analyze how **inefficient** standard search algorithms like Genetic Programming were under limited budget, showing how often they revisited the same expressions <d-cite bibtex-key="kronberger2024inefficiency"></d-cite>.

At this point, there was much more we could do with e-graphs in SR...

## Generating uniqueness

In a recent work on **e-graph genetic programming** ([**eggp**](https://github.com/folivetti/eggp)) <d-cite bibtex-key="de2025improving"></d-cite>, the authors turned the e-graph into a **database and guidance system** for equation discovery.

Remember how genetic programming works:
- Create initial random expressions
- Repeat:
    * Select two expressions proportional to their performance
    * Combine parts of these expressions generating a new expression
    * Replace a part of this expression with a random variation

As stated before, this can be inefficient since we can generate many equivalent expressions during the process <d-cite bibtex-key="kronberger2024inefficiency"></d-cite>. But, what if we store every generated expression into a single e-graph and run the equality saturation algorithm?

For once, we would have a database system allowing us to query whether a given expression was already visited, even in an equivalent form. But also, we can use this information to enforce the generation of new expressions!

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/blog6.png" class="img-fluid" %}

It works like this, imagine that the current state of the search is the e-graph above! The green e-classes are the root of the already evaluated expressions.
Let's say that GP decides to recombine the expressions $x + \sqrt{x}$ and $x + 2x$, choosing to replace $\sqrt{x}$ of the first expression with something else from the second.
The choices of recombination are $\{x+x, x+2, x+2x, x+x+2x\}$. We can query each one of these choices to verify whether they already exist in the e-graph. If they do and were already evaluated, we discard them!

Similarly, we can do the same for the mutation. let's suppose we will mutate the expression $x + \sqrt{x}$ by replacing $\sqrt{x}$ with a random expression. If we are unlucky, we may generate the expression $2x$, thus forming $x+2x$, which was already evaluated. After detecting the duplicate, we can change the multiplication in $2x$ with any binary operator that would generate a new expression!

## Explore the Search Space with `eggp` and `rEGGression`

You can start using this algorithm right now! Here is how you can install the library and use it to find an equation for a real-world fluid dynamics dataset.
You can install eggp with `pip`:

```bash
pip install eggp
```

### Finding a Formula 
This example uses `eggp` to find a relationship for fluid dynamics data the `nikuradse_1.csv` dataset (see the tutorials at this [link](https://github.com/folivetti/eggp/tree/main/tutorials)).

```python
from eggp import EGGP
import pandas as pd 

pd.set_option('display.max_colwidth', 100)
df = pd.read_csv("datasets/nikuradse_1.csv")

model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', simplify=True, dumpTo='regression_example.egg')

model.fit(df[['r_k', 'log_Re']], df['target'])
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])
```

After running the search, the final e-graph (stored in `regression_example.egg`) contains the entire history of visited, unique solutions.  This can be used to resume the search with  different settings, such as a different nonterminal set:

```python
model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log,sin,tanh', loss='MSE', loadFrom='regression_example.egg')

model.fit(df[['r_k', 'log_Re']], df['target'])

print("\nLast population resumed from the first Pareto front: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])
```
### Interactive Model Selection with `rEGGression`

This e-graph can be further explored with the [rEGGression](https://github.com/folivetti/reggression) tool <d-cite bibtex-key="de2025reggression"></d-cite>. An e-graph explorer for Symbolic Regression.

```python
from reggression import Reggression

egg = Reggression(dataset="datasets/nikuradse_1.csv", loadFrom="regression_example.egg", loss="MSE") 
print(egg.top(5, pattern="v0 ^ v0")
```

This will retrieve the top 5 expressions that follow the pattern $\alpha^\alpha$, such as $x^x$ or $\log((x+5)^{x+5}) + 3$. The result is a list of the best-performing models matching your structural criteria:

| Expression | Fitness | Size |
|---------------|--------|----|  
| $\left({\operatorname{log}({log_{Re}^{log_{Re}}})^{\theta_{0}}} \cdot r_{k}\right)^{\theta_{1}}$ | -0.001514 | 10 |  
| $\left(\left({log_{Re}^{log_{Re}}} \cdot \theta_{0}\right) + \frac{\theta_{1}}{\operatorname{log}(r_{k})}\right)$ | -0.001567 | 10  |
| $\left(\frac{\operatorname{log}({log_{Re}^{log_{Re}}})}{\left(\theta_{0} \cdot r_{k}\right)} + \theta_{1}\right)$ | -0.004623 | 10  |
| $\left(\frac{\left(r_k + \theta_0\right)^{\theta_1}}{\log(r_k)^{\log(r_k)}} + \theta_2\right)$ | -0.005701 | 13  |
| $\left(\operatorname{log}({log_{Re}^{log_{Re}}}) \cdot r_{k}\right)^{\theta_{0}}$ | -0.010011 | 8 |

Or retrieving the top-5 expressions **not** having the pattern $\log(v)$:

```python
print(egg.top(5, pattern="log(v0)", negate=True)
```
| Expression | Fitness | Size |
|---------------|--------|----| 
| $\left(\left(\theta_0 \cdot r_k\right)^{\theta_1}^{log_{Re}} \cdot \theta_2\right)$ | -0.001131 |11  |
| $\left({\left(log_{Re} \cdot \theta_{0}\right)^{\theta_{1}}} \cdot \left(r_{k} + \theta_{2}\right)\right)^{\theta_{3}}$ |-0.001187 |11  |
| \left(\frac{\left(r_k + \theta_0 \right)^{\theta_1}}{\left(\frac{\theta_2}{log_{Re}} + \theta_3\right)} + \theta_4\right)$ |-0.001190| 13  |
| $\left({\left(e^{\left(log_{Re} + \theta_{0}\right)} \cdot \theta_{1}\right)^{\theta_{2}}} \cdot r_{k}\right)^{\theta_{3}}$ | -0.001191 |12  |
| $\left(\theta_0 \cdot \left(\left(\left(log_{Re} \cdot log_{Re}\right) \cdot \theta_{1}\right) + r_{k}\right)\right)^{\theta_{2}}$ | -0.001192 |11|

### Benchmark 

Running `eggp` and other SotA algorithms from the literature in a selection of real-world datasets, we can see that `eggp`stands out among the best algorithms, given the statistical test:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/rank_mse.png" class="img-fluid" %}

And promotes the smaller models among the top-performant algorithms:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/rank_size.png" class="img-fluid" %}

A more detailed pairwise comparison can be seen with a BBT plot:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/bbt.png" class="img-fluid" %}

## A Powerful Database for Equations: Using e-graphs and Equality Saturation for Interactive Equation Discovery

For this experiment, we will use a variation of one of the benchmarks proposed by E. J. Vladislavleva et al<d-cite bibtex-key="vladislavleva2008order"></d-cite>:

$$
e^{-x/1.2}\, x^3  \left(\cos(x)\, \sin(x)^2 - 3.1415\right)
$$

Let's generate data points in the range $[0, 10]$ while adding a bit of Gaussian noise:

```python
x = np.arange(0, 10, 0.05)
y = np.exp(-x/1.2)*x**3*(np.cos(x) \
    * np.sin(x)**2 - 3.1415) \
    + np.random.normal(0, 0.05, x.shape)
```

To make things a bit more interesting for this post, we will use just the middle part for training and the rest as a test set:

```python
lb, ub = 2.1, 5
x_sel = x[(x>lb) & (x<ub)].reshape(-1,1)
y_sel = y[(x>lb) & (x<ub)]
x_ood = x[(x<=lb) | (x>=ub)].reshape(-1, 1)
y_ood = y[(x<=lb) | (x>=ub)]
```

Plotting the training set as red dots and the test set as green dots, we have:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/vlad.svg" class="img-fluid" %}


We are, of course, making things harder for symbolic regression:
1. The relationship is nonlinear.
2. The training set is insufficient to guarantee an unique global optima.

In any case, the purpose here is to show how we can use r🥚ression to explore alternative models.

## Laying the egg 🥚

We can create an initial e-graph for this dataset using `eggp`. As mentioned [in the previous post](https://symreg.at/blog/2025/equality-saturation-and-symbolic-regression/), this algorithm uses e-graphs to enforce the generation of new expressions, avoiding redundancy in the search.

```python
from eggp import EGGP
import pandas as pd

reg = EGGP(gen=200, nPop=200, maxSize=25, \
      nonterminals="add,sub,mul,div,log,power,sin,cos,abs,sqrt", \
      simplify=True, optRepeat=2, optIter=20, folds=2,  \
      dumpTo="vlad.egg")
reg.fit(x_sel, y_sel)
```

Some observations:
- The non-terminal set is large in order to generate many different alternative models.
- We are not running for a large number of iterations, so we could possibly find better models with proper settings.
- The maximum size is larger than the true equation.

We are saving the final e-graph into the file named `vlad.egg` so we can explore it after the search.
Looking at the results we can see the Pareto front with different trade-offs of accuracy and size.

| Math                                                                                                                                                                                                                                 |   size |   loss_train |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|-------------:|
| $$\theta_{0}$$                                                                                                                                                                                                                       |      1 |   0.360495   |
| $$\left(\theta_{0} + \text{cos}(x_{0})\right)$$                                                                                                                                                                              |      4 |   0.319114   |
| $$\left(\theta_{0} + \frac{\theta_{1}}{x_{0}}\right)$$                                                                                                                                                                               |      5 |   0.318433   |
| $$\left(\theta_{0} + (\left(\theta_{1} - x_{0}\right))^2\right)$$                                                                                                                                                 |      6 |   0.0641624  |
| $$\left(\left(\text{cos}(x_{0}) \cdot \left(x_{0} + \theta_{0}\right)\right) + \theta_{1}\right)$$                                                                                                                           |      8 |   0.0421559  |
| $$\left(\theta_{0} + \left(\text{cos}(\left(\theta_{1} + \text{cos}(x_{0})\right)) \cdot x_{0}\right)\right)$$                                                                                                       |      9 |   0.012507   |
| $$\left(\theta_{0} + \left(\text{cos}(\left(\theta_{1} + \text{cos}(x_{0})\right)) \cdot \left(\theta_{2} \cdot x_{0}\right)\right)\right)$$                                                                         |     11 |   0.00899634 |
| $$\left(\theta_{0} + \left(\text{cos}(\text{cos}(x_{0})) \cdot \left(\theta_{1} \cdot \text{cos}(\left(x_{0} + \theta_{2}\right))\right)\right)\right)$$                                                     |     12 |   0.0046806  |
| $$\left(\theta_{0} - \left(\text{cos}(\text{cos}(x_{0})) \cdot \left(\theta_{1} \cdot \text{cos}(\left (\left(x_{0} + \theta_{2}\right)\right ))\right)\right)\right)$$                                      |     13 |   0.00481255 |
| $$\left(\theta_{0} + \left(\text{cos}(\text{cos}(x_{0})) \cdot \left(\theta_{1} \cdot \text{cos}(\left(\left(\theta_{2} - x_{0}\right) + \theta_{3}\right))\right)\right)\right)$$                           |     14 |   0.00586963 |
| $$\left(\left(\text{cos}(\text{cos}(x_{0})) \cdot \left(\theta_{0} \cdot \text{cos}(\left(\left(\theta_{1} - \left(x_{0} + \theta_{2}\right)\right) + \theta_{3}\right))\right)\right) + \theta_{4}\right)$$ |     16 |   0.00547237 |


## Hatching the egg 🐣

Now, let's load the e-graph into `r🥚ression`:

```python
from reggression import Reggression
egg = Reggression(dataset="vlad.csv", loadFrom="vlad.egg")
```

If we look at the top-5 models, we can see small variations of the top performing with similar fitness (negative MSE) values.

```python
egg.top(5)[["Latex", "Fitness", "Size"]]
```

| Latex                                                                                                                                                                                                                                                                      |     Fitness |   Size |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|-------:|
| $$\left(\left(\text{cos}(\text{cos}(x)) \cdot \left(\theta_{0} \cdot \text{cos}(\left(\left(\theta_{1} - \left(x + \theta_{2}\right)\right) + \theta_{3}\right))\right)\right) + \theta_{4}\right)$$                           | -0.00415306 |     16 |
| $$\left(\theta_{0} + \left(\text{cos}(\text{cos}(x)) \cdot \left(\text{cos}(\left(\mid\mid\left(\left(x + \theta_{1}\right) + \theta_{2}\right)\mid\mid + \theta_{3}\right)) \cdot \theta_{4}\right)\right)\right)$$ | -0.00425244 |     18 |
| $$\left(\left(\left(\text{cos}(\text{cos}(x)) \cdot \left(\text{cos}(\left(\left(\theta_{0} - \left(x + \theta_{1}\right)\right) + \theta_{2}\right)) \cdot \theta_{3}\right)\right) + \theta_{4}\right) + \theta_{5}\right)$$ | -0.00430326 |     18 |
| $$\left(\theta_{0} + \left(\left(\text{cos}(\text{cos}(x)) \cdot \left(\text{cos}(\left(\mid\left(\sqrt{x}^2 + \theta_{1}\right)\mid + \theta_{2}\right)) \cdot \theta_{3}\right)\right) + \theta_{4}\right)\right)$$     | -0.00430774 |     19 |
| $$\left(\theta_{0} + \left(\text{cos}(\text{cos}(x)) \cdot \left(\theta_{1} \cdot \text{cos}(\left(\left(\theta_{2} - x\right) + \theta_{3}\right))\right)\right)\right)$$                                                     | -0.0043503  |     14 |

Some of these functions behave similarly while others display a different behavior when looking outside of the training region:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/top5.svg" class="img-fluid" %}

We can also plot the best models while limiting the maximum size:

```python
model_top(egg.top(n=10, filters=["size <= 10"]), n, x, y)
```
{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/top10.svg" class="img-fluid" %}

We can see even more different behaviors compared to the previous plot but, sill, none of them are even close to the correct one  :-( 

Since we are still far from the true expression, let us investigate the distribution of the tokens of the top 1000 generated expressions. 

```python
egg.distributionOfTokens(top=1000)
```
This command returns a table with the number of times each token was used in the top expressions and the average fitness of the expressions that contains such token.  The table is ordered by average fitness (negative MSE).

| Pattern    |   Count |      AvgFit |
|:-----------|--------:|------------:|
| x0         |    2604 | -0.00359749 |
| t0         |    1006 | -0.009312   |
| t1         |     981 | -0.00941213 |
| t2         |     955 | -0.00937039 |
| t3         |     806 | -0.00893546 |
| t4         |     466 | -0.00910986 |
| t5         |     144 | -0.00786632 |
| t6         |       1 | -0.013187   |
| Abs(v0)    |     465 | -0.00810496 |
| Sin(v0)    |      74 | -0.0115615  |
| Cos(v0)    |    3029 | -0.00309273 |
| Sqrt(v0)   |      32 | -0.00845579 |
| Square(v0) |      27 | -0.00967352 |
| Log(v0)    |      10 | -0.00972384 |
| Exp(v0)    |      45 | -0.0118458  |
| Cube(v0)   |      38 | -0.00867039 |
| (v0 + v1)  |    3405 | -0.00275121 |
| (v0 - v1)  |     351 | -0.00848634 |
| (v0 * v1)  |    2139 | -0.0042415  |
| (v0 / v1)  |      68 | -0.00815694 |

Apart from the first rows that displays the terminals, we can see that the absolute value function is frequently used and often contributes to a lower fitness, even though it is not present in the ground-truth expression.

> When we have partial functions such as `log` and `sqrt`, the absolute value can help "fixing" invalid inputs.

Sine and cosine are ranked next, but with cosine being more often used. The exponential is rarely used and particularly with a worse average fitness than the other tokens. The reason for this could be that fitting parameters inside an exponential function can be tricky depending on the initial values.

We can verify that by plotting the top 5 expressions with the pattern $e^{\square_0}\square_1$ with the command:

```python
egg.top(n=n, pattern="exp(v0)*v1")
```

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/top5pat.svg" class="img-fluid" %}

as we can see, still not a very good fit, as expected.


## With a little help from my friends 🐣🐤

We can try our luck with another SR method, such as Operon <d-cite bibtex-key="operon"></d-cite>, and insert the obtained expressions into the e-graph:

```python
from pyoperon.sklearn import SymbolicRegressor
regOp = SymbolicRegressor(objectives=['mse','length'], max_length=20, allowed_symbols='add,sub,mul,div,square,sin,cos,exp,log,sqrt,abs,constant,variable')
regOp.fit(x_sel, y_sel)
f = open("equations.operon", "w")
for eq in regOp.pareto_front_:
  eqstr = regOp.get_model_string(eq['tree'])
  fitness = -eq['mean_squared_error']
  print(f"{eqstr},{fitness},{fitness}", file=f)
f.close()
egg.importFromCSV("equations.operon")
```
Plotting the top-5 expressions we get:

{% include figure.liquid path="assets/img/2026-04-27-egraph-symreg/top5operon.svg" class="img-fluid" %}

Still no luck! But we didn't make things easy for SR anyway!

We can insert the ground-truth expression to see whether the parameter optimization is capable of converging to the true parameters and if the fitness is better than what we have.

```python
egg.insert("exp(x0/t0)*(x0^3)*(cos(x0)*(sin(x0)^2)-t1)")
```

| Latex                                                                                                                                                                                                        |     Fitness | Parameters                                |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|:------------------------------------------|
| $$\left(\left({x^{3.0}} \cdot \left(\left(\text{cos}(x) \cdot {\text{sin}(x)^{2.0}}\right) + \theta_{0}\right)\right) \cdot e^{\left(x \cdot \theta_{1}\right)}\right)$$ | -0.00256414 | [-3.15, -0.83] |

The answer is YES! We can get the ground-truth expression with enough iterations and a larger amount of luck :-)
Or, we can even resort to adding some constraints <d-cite bibtex-key="shapeconstraints"></d-cite>...


### It's all the same, no matter where you are 🐥🐥🐥

We can also use r🥚ression to check whether two or more expressions are equivalent. Let's say we want to see whether $(x+3)^2 - 9$ and $x(x + 6)$ are the same. 

First, we create an empty e-graph:

```python
newegg = Reggression(dataset="vlad.csv", loss="MSE")
```

Next, we add both expressions while storing their e-class ids:

```python
eid1 = egg.insert("(x0 + 3)**2 - 9").Id.values[0]
eid2 = egg.insert("x0*(x0 + 6)").Id.values[0]
print(eid1, eid2)
> 6, 9
```

Initially, their ids are going to be different, since until now they are distinct to each other as far as the e-graph is concerned.

Now, the main idea is that we run equality saturation to produce all the equivalent forms of each one of these expressions following a set of rules, such as:

$$
(x + y)^2 \rightarrow x^2 + y^2 + 2xy
$$

> If the set of rules are sufficient to produce at least one common expression departing from the first and from the second expressions, they will eventually be merged, and their e-class id will become the same.

We can run some iterations of equality saturation using the command:

```python
egg.eqsat(5)
```

And, now, their ids should be the same!

```python
print("Id of the first equation: \n", egg.report(eid1).loc[0:1, ["Info", "Training"]])
print("Id of the second equation: \n", egg.report(eid2).loc[0:1, ["Info", "Training"]])
> Id of the first equation: 16
> Id of the second equation: 16
```

After running equality saturation, we can also retrieve a sample of the equivalent expressions for that e-class id:

```python
egg.getNExpressions(eid1, 10)
```

Leading to:

$$((6.0 + x) * x)$$
$$((x + 6.0) * x)$$
$$((x * 6.0) + (x ^ 2))$$
$$((x * 6.0) + (x ^ 2))$$
$$(0.0 + ((6.0 + x) * x))$$
$$(0.0 + ((x + 6.0) * x))$$
$$((2.0 * (x * 3.0)) + (x ^ 2))$$
$$((2.0 * (3.0 * x)) + (x ^ 2))$$
$$(((x * 3.0) * 2.0) + (x ^ 2))$$
$$(((3.0 * x) * 2.0) + (x ^ 2))$$

This can potentially be used to integrate e-graphs with other genetic programming algorithms or even reward based algorithms such as Monte Carlo Tree Search <d-cite bibtex-key="kamienny2023deep"></d-cite> <d-cite bibtex-key="sun2022symbolic"></d-cite> and Deep Reinforcement Learning <d-cite bibtex-key="mundhenk2021symbolic"></d-cite>, and LLMs <d-cite bibtex-key="shojaee2024llm"></d-cite>.

### Technical Details

The e-graph implementation is available at the [Haskell Symbolic Regression](https://github.com/folivetti/srtree) library with some differences from [egg](https://docs.rs/egg/latest/egg/) to make it more convenient for symbolic regression and memory efficient.

The SR algorithm [eggp](https://github.com/folivetti/eggp) already shows the potential of this integration, being capable of beating the state-of-the-art with a simple genetic programming framework.

The  [rEGGression](https://github.com/folivetti/reggression) Python library make it easy to explore the explored solutions and can be used as an interactive tool for a guided model selection. 
