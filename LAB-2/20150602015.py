import csv
import numpy as np
import matplotlib.pyplot as plt


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


exp_list_1 = np.array([])
exp_list_2 = np.array([])

sal_list_1 = np.array([])
sal_list_2 = np.array([])

temp1 = np.array([])  # Experience
temp2 = np.array([])  # Salary

with open("team_""big.csv") as f:
    All = list(csv.reader(f))

for row in All:
    if row != All[0]:
        temp1 = np.append(temp1, int(row[6]))  # Experience
        temp2 = np.append(temp2, int(row[8]))  # Salary

exp_list_1 = np.array_split(temp1, 2)[0]
exp_list_1 = exp_list_1.astype(np.double)

exp_list_2 = np.array_split(temp1, 2)[1]
exp_list_2 = exp_list_2.astype(np.double)

sal_list_1 = np.array_split(temp2, 2)[0]
sal_list_1 = sal_list_1.astype(np.double)

sal_list_2 = np.array_split(temp2, 2)[1]
sal_list_2 = sal_list_2.astype(np.double)


def simlin_coef(x, y):
    avg_x = avg(x)
    avg_y = avg(y)
    counter = 0
    result = np.array([])
    up = 0.0
    down = 0.0

    while counter < x.size:
        up += (x[counter] - avg_x) * (y[counter] - avg_y)
        down += square((x[counter] - avg_x))
        counter = counter + 1

    a = up / down  # B1
    b = avg_y - a * avg_x  # B0
    result = np.append(result, a)
    result = np.append(result, b)
    return result


def simlin_plot(x, y, a, b):
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

    return result_set


def new_function(x, y):
    counter = 0
    rss = 0.0
    tss = 0.0
    avg_x = avg(x)
    while counter < len(x):
        rss += square(x[counter] - y[counter])
        tss += square(x[counter] - avg_x)
        counter += 1
    r2 = 1 - (rss / tss)
    print(r2)


first_results = simlin_coef(exp_list_1, sal_list_1)
second_results = simlin_coef(exp_list_2, sal_list_2)

b0_1 = first_results[0]
b1_1 = first_results[1]

b0_2 = second_results[0]
b1_2 = second_results[1]

print(first_results[0], first_results[1])
print(second_results[0], second_results[1])

sal_pred_1 = simlin_plot(exp_list_1, sal_list_1, b0_2, b1_2)
sal_pred_2 = simlin_plot(exp_list_2, sal_list_2, b0_1, b1_1)

new_function(sal_list_1, sal_pred_1)
new_function(sal_list_2, sal_pred_2)

print(b0_2, b1_2)

plt.show()
