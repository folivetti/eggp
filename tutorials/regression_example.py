from eggp import EGGP
import pandas as pd 

pd.set_option('display.max_colwidth', 100)
# Load the example dataset 
df = pd.read_csv("datasets/nikuradse_1.csv")

# Create an EGGP object, all parameters have default values. We will dump the results to a file named 'regression_example.egg'
# gen is the number of generations, nPop is the population size, maxSize is the maximum size of the expressions 
# nTnament is the tournament size, pc and pm are the probabilities for performing crossover and mutation 
# nonterminals is the function set 
# loss is the loss function (see the boston house example for other values)
# optIter and optRepeat are the number of iterations and repetitions for the optimization of constants
# nParams is the number of parameters to include in the expression 
# max_time is the maximum time in seconds to run the algorithm
# simplify indicates whether to simplify the expressions using equality saturation 
# dumpTo dumps the final e-graph to a file
# By default it will run a multi-objective approach.
model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=False, dumpTo='regression_example.egg')

model.fit(df[['r_k', 'log_Re']], df['target'])

print("Multi-objective mode returning the Pareto front without simplifcation: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])

# Running with simplification enabled. Notice that the size used to determine the pareto front is before simplification.
# It is expected to see two elements on the front with the same size.
model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=True)

model.fit(df[['r_k', 'log_Re']], df['target'])

print("\nPareto front with simplifcation enabled: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])

# Let's try the generational approach, where the entire population is replaced by the offspring at each generation.
model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=True, generational=True)

model.fit(df[['r_k', 'log_Re']], df['target'])

print("\nLast population of generational approach with simplifcation: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])

# 'loadFrom' argument enable us to start the search from a previously saved search.
# Notice that the egg file will contain ALL the evaluated expressions, not only the final population.
# As the initial population it will choose either the top N solutions (if generational) or the Pareto front (if MOO).
model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=True, generational=True, loadFrom='regression_example.egg')

model.fit(df[['r_k', 'log_Re']], df['target'])

print("\nLast population resumed from the first Pareto front: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])
