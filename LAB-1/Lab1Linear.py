import csv
import numpy as np
import matplotlib.pyplot as plt

Experience = np.array([])  # X Vector for analysis
Salary = np.array([])  # Y Vector for analysis

with open("team_big.csv") as f:
    lst = list(csv.reader(f))  # ------- Read csv file in to the array

for row in lst:
    if row != lst[0]:  # Skip Column headers
       Experience = np.append(Experience, int(row[6]))  # ------- put experience vector into numpy array
       Salary = np.append(Salary, int(row[8]))  # --------- the salary vector next


def simlin_coef(x, y):  # -------- 50 pts. function def.
    x_av = np.mean(x)
    y_av = np.mean(y)
    outp = np.array([])
    nom = 0.0
    denom = 0.0
    index = 0   
    while index < x.size:
          nom += (x[index] - x_av) * (y[index] - y_av)
          denom += (x[index] - x_av) * (x[index] - x_av)  # B1
          index += 1
    a = nom / denom
    b = y_av - a * x_av  # B0
    outp = np.append(outp, a)
    outp = np.append(outp, b)
    return outp

def simlin_plot(x, y, a, b):  # ------ 30 pts. function def.
    index = 0
    outp_set = np.array([])
    while index < x.size:
          outp_set = np.append(outp_set, a * x[index] + b)
          index += 1
    plt.figure()
    print(outp_set)
    plt.plot(x, outp_set, color="r")
    plt.scatter(x, y, color="b")
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.show()

finalres = simlin_coef(Experience, Salary)

simlin_plot(Experience, Salary, finalres[0], finalres[1])
