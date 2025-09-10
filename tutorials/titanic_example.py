from eggp import EGGP
import pandas as pd 
import numpy as np

pd.set_option('display.max_colwidth', 100)
# Load the example dataset 
df = pd.read_csv("datasets/Titanic-Dataset.csv")

# In this dataset the target variable is 'Rented Bike Count' which is usually modelled as a Poisson regression.
# We will first fit using MSE as the loss function and compare with the Poisson regression later.
model = EGGP(gen=100, nPop=100, maxSize=20,  nonterminals='add,sub,mul,div,power,exp,log', loss='Bernoulli', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=True)

model.fit(df[['Pclass', 'Fare', 'Age']], df['Survived'])

print("Titanic Bernoulli loss: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])

yhat = model.predict(df[['Pclass', 'Fare', 'Age']])
acc = np.sum(np.round(yhat) == df['Survived']) / yhat.shape  # Check accuracy
print("Accuracy: ", acc)
