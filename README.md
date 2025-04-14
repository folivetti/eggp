# pyeggp - Python e-graph GP

Python bindings for [eggp](https://github.com/folivetti/srtree/blob/main/apps/eggp/README.md).

ggp (e-graph genetic programming), follows the same structure as the traditional GP. The initial population is created using ramped half-and-half respecting a maximum size and maximum depth parameter and, for a number of generations, it will choose two parents using tournament selection, apply the subtree crossover with probability $pc$  followed by the subtree mutation with probability $pm$, when the offsprings replace the current population following a dominance criteria.

How to install the package:

```bash
pip install pyeggp
```

The bindings were created following the amazing example written by [wenkokke](https://github.com/wenkokke/example-haskell-wheel)
