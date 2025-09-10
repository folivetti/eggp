from eggp import EGGP
import pandas as pd 

pd.set_option('display.max_colwidth', 100)
# Load the example dataset 
df = pd.read_csv("datasets/SeoulBikeData.csv")

# In this dataset the target variable is 'Rented Bike Count' which is usually modelled as a Poisson regression.
# We will first fit using MSE as the loss function and compare with the Poisson regression later.
model = EGGP(gen=100, nPop=100, maxSize=15,  nonterminals='add,sub,mul,div,power,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=True)

model.fit(df[['Hour','Temp','Hum','WindSp','Vis','Dewpoint','SolarRad','Rainfall','Snowfall']], df['bikes'])

print("Bike rentals with MSE: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])

# Searching for a Poisson model. Notice that the data must not contain 0s in the target variable for Poisson regression.
model = EGGP(gen=100, nPop=100, maxSize=15,  nonterminals='add,sub,mul,div,power,exp,log', loss='Poisson', optIter=100, optRepeat=5, nParams=2, folds=2, max_time=120, simplify=True)

model.fit(df[['Hour','Temp','Hum','WindSp','Vis','Dewpoint','SolarRad','Rainfall','Snowfall']], df['bikes'])

print("\nBike rentals with Poisson: ")
print(model.results[['Expression', 'loss_train', 'loss_val', 'size']])

