import csv
import numpy as np
import matplotlib.pyplot as plt

Experience = np.array([])  # our X
Salary = np.array([])  # our Y

with open("team_big.csv") as f:
    Total = list(csv.reader(f))  # append all csv info to the array

for row in Total:
    if row != Total[0]:  # first row contains references so we pass them
        Experience = np.append(Experience, int(row[6]))  # append the experience values to the numpy array
        Salary = np.append(Salary, int(row[8]))  # append the salary info to the next array


def avg(c):  # For finding the average of a numpy array
    c = np.asarray(c)
    counter = 0
    summary = 0.0
    while counter < len(c):
        summary += c[counter]
        counter += 1
    average = summary / len(c)
    return average


def square(sqr):
    result = (sqr * sqr)
    return result


def simlin_coef(x, y):  # 50 point function definition
    avg_x = avg(x)
    avg_y = avg(y)
    counter = 0
    result = np.array([])
    up = 0.0
    down = 0.0

    while counter < x.size:
        up += (x[counter] - avg_x) * (y[counter] - avg_y)
        down += square((x[counter] - avg_x))  # B1
        counter = counter + 1

    a = up / down
    b = avg_y - a * avg_x  # B0
    result = np.append(result, a)
    result = np.append(result, b)
    return result


def simlin_plot(x, y, a, b):  # 30 point function definition
    counter = 0
    result_set = np.array([])

    while counter < x.size:
        result_set = np.append(result_set, a * x[counter] + b)
        counter += 1

    plt.figure()
    plt.plot(x, result_set, color="red")
    plt.scatter(x, y, color="blue")
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.show()


results = simlin_coef(Experience, Salary)

print(results[0], results[1])

simlin_plot(Experience, Salary, results[0], results[1])
