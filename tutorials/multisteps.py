from eggp import EGGP
import pandas as pd 
import numpy as np 

pd.set_option('display.max_colwidth', 100)
# Load the example dataset 
df = pd.read_csv("datasets/nikuradse_1.csv")

# split dataframe into multiple sets based on column r_k 
Xs, ys, r_ks = [], [], [] 
for r_k_value, group in df.groupby('r_k'):
    Xs.append(group[['log_Re']].values)
    ys.append(group['target'].values)
    r_ks.append(r_k_value)

model = EGGP(gen=200, nPop=500, maxSize=50, nTournament=3, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,power', loss='MSE', optIter=100, optRepeat=5, nParams=3, folds=1, simplify=True)

# We fit a "multi-view" assuming each value of r_k is a view 
# Since in this dataset we have a limited number of unique r_k values,
# we can think of it as finding an expression f(log_re; theta) where theta = g(r_k)
# We are limiting to one parameter for simplicity.
model.fit_mvsr(Xs, ys)

# Let's get the best model and print the max-loss 
best_ix = model.results.maxloss.idxmin()
best_id = model.results.loc[best_ix, 'id']

best_df = model.results[model.results['id'] == best_id]

n_thetas = len(best_df.iloc[0]['theta'].split(";"))
thetas = best_df.theta.values 

print('Best expression: ', best_df.iloc[0]['Expression'])
print('MSE of each view: ', best_df.loss_train.values)

# For each parameter (if we have more than one), fit a model to create an expression theta(r_k)
for i in range(n_thetas):
    ts = list(map(lambda x: float(x.split(";")[i]), thetas))
    print(f"Theta {i} values across different views: {ts}")
    model_v = EGGP(gen=200, nPop=500, maxSize=15, nTournament=3, pc=0.8, pm=0.2, nonterminals='add,sub,mul,div,exp,log', loss='MSE', optIter=100, optRepeat=5, nParams=-1, folds=1, simplify=True)
    model_v.fit(np.array(r_ks).reshape(-1,1), np.array(ts))
    best_ix_v = model_v.results.loss_val.idxmin()
    # The loss should be 0, otherwise the error will propagate to the main model
    print(f"Simplified expression for theta {i}: {model_v.results.loc[best_ix_v, 'Expression']}")
    print(f"MSE of theta(r_k) model, this should be 0 otherwise the error propagates: {model_v.results.loc[best_ix_v, 'loss_train']}")
