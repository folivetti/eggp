from pyeggp import pyeggp_run

args = ["test/data.csv", "100", "10", "1", "MSE", "50", "2", "-1", "100", "3", "0.9", "0.3", "add,sub,mul,div", "\"\"", "\"\"", "True"]
output = pyeggp_run("test/data.csv", 100, 100, 10, 3, 0.9, 0.3, "add,sub,mul,div", "MSE", 50, 2, -1, 3, 0, "", "")

print(output)
