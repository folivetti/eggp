from eggp import eggp_run, EGGP
import pandas as pd

output = eggp_run("test/data.csv", 100, 100, 10, 3, 0.9, 0.3, "add,sub,mul,div,log", "Gaussian", 50, 2, -1, 3, 0, 0, 0, 0, "", "", "", 0)

print(output)

output = eggp_run("test/data.csv test/data2.csv", 100, 100, 10, 3, 0.9, 0.3, "add,sub,mul,div,log", "Gaussian", 50, 2, -1, 3, 0, 0, 0, 0, "", "", "", 0)

print(output)

print("Check EGGP")
df = pd.read_csv("test/data.csv")
Z = df.values
X = Z[:,:-1]
y = Z[:,-1]

reg = EGGP(gen=100, nPop=100, maxSize=10, nTournament=3, pc=0.9, pm=0.3, nonterminals="add,sub,mul,div,log", loss="Gaussian", optIter=50, optRepeat=2, nParams=-1, folds=3, max_time=0, simplify=False, trace=False, generational=False, dumpTo="", loadFrom="", useFracBayes=False)
reg.fit(X, y)
print(reg.score(X, y))

reg.fit_mvsr([X,X],[y,y])
print(reg.predict_mvsr(X,0))
print(reg.predict_mvsr(X,1))
print(reg.results)
