from eggp import EGGP
import pandas as pd 

pd.set_option('display.max_colwidth', 100)
# Load the example dataset 
df0 = pd.read_csv("datasets/example0.csv")
df1 = pd.read_csv("datasets/example1.csv")
df3 = pd.read_csv("datasets/example3.csv")

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
model = EGGP(gen=100, nPop=100, maxSize=15, nTournament=5, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=1, folds=1, max_time=120, simplify=False, dumpTo='regression_example.egg')

model.fit_mvsr([df[['Xaxis0']] for df in [df0,df1,df3]], [df['yaxis'] for df in [df0,df1,df3]])

print("Multi-objective mode returning the Pareto front without simplifcation: ")
print(model.results[["Expression", "view", "loss_train", "maxloss"]])
