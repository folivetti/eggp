from eggp import eggp_run, EGGP
import pandas as pd

output = eggp_run("test/data.csv", 100, 100, 10, 3, 0.9, 0.3, "add,sub,mul,div,log", "MSE", 50, 2, -1, 3, 0, 0, 0, "", "")

print(output)

output = eggp_run("test/data.csv test/data2.csv", 100, 100, 10, 3, 0.9, 0.3, "add,sub,mul,div,log", "MSE", 50, 2, -1, 3, 0, 0, 0, "", "")

print(output)

print("Check EGGP")
df = pd.read_csv("test/data.csv")
Z = df.values
X = Z[:,:-1]
y = Z[:,-1]

reg = EGGP(100, 100, 10, 3, 0.9, 0.3, "add,sub,mul,div,log", "LOG10", 50, 2, -1, 3, True, False, False, "", "")
reg.fit(X, y)
print(reg.score(X, y))

reg.fit_mvsr([X,X],[y,y])
print(reg.predict_mvsr(X,0))
print(reg.predict_mvsr(X,1))
print(reg.results)
