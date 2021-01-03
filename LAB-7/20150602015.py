import csv
import numpy as np
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.set_printoptions(suppress=True)

Age = np.array([])  # Input X #
Length = np.array([])  # Input Y #
Matrix = np.array([])


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


def calculate_coefficients(x, y):
    trans = np.transpose(x)
    first = np.dot(trans, x)
    second = np.linalg.inv(first)
    third = np.dot(second, trans)
    result = np.dot(third, y)
    return result


def calculate_r2(x, y):
    counter = 0
    rss = 0.0
    tss = 0.0
    avg_x = avg(x)
    while counter < len(x):
        rss += square(x[counter] - y[counter])
        tss += square(x[counter] - avg_x)
        counter += 1
    r2 = 1 - (rss / tss)
    return r2


def cubic_spline_regression(x, y, d):
    ones = np.ones((len(x), 1))  # initialize Ones array #
    temp = np.array([])
    counter = 0

    while counter < d:
        counter += 1
        temp_x = pow(x, counter)

        if counter == 1:
            temp = np.column_stack((ones, temp_x))
        else:
            temp = np.column_stack((temp, temp_x))

    B = calculate_coefficients(temp, y)
    Y = np.dot(temp, B)
    return Y


with open("Bluegill_dataset.csv") as f:
    Total = list(csv.reader(f))

for row in Total:
    if row != Total[0]:
        Age = np.append(Age, int(row[0]))
        Length = np.append(Length, int(row[1]))

Ones = np.ones((len(Age), 1))
Matrix = np.column_stack((Ones, Age))
B_1 = calculate_coefficients(Matrix, Length)
Y_1 = np.dot(Matrix, B_1)
R2_1 = calculate_r2(Length, Y_1)

print("Linear Regression R2 Score: ", R2_1)

results = np.column_stack((Y_1, Age, Length))
counter_2 = 1
Total_Matrix = np.array([])

while counter_2 < 6:
    counter_2 += 1
    Y_2 = cubic_spline_regression(Age, Length, counter_2)
    R_2 = calculate_r2(Length, Y_2)
    results = np.column_stack((results, Y_2))
    print("R2 values for the power of ", counter_2, ":", R_2)

sorted_indices = np.argsort(Age)
results = results[sorted_indices]  # changed the indices as you requested #


plt.figure("20150602015")
plt.scatter(Age, Length, color="blue")

plt.title("Polynomial Regression")
plt.xlabel("Age of Bluegill Fish")
plt.ylabel("Length of Bluegill Fish")

plt.plot(results[:, [1]], results[:, [0]], "green", label="Linear")
plt.plot(results[:, [1]], (results[:, [5]]), "red", label="3rd Degree")
plt.plot(results[:, [1]], (results[:, [6]]), "yellow", label="5th Degree")
plt.plot(results[:, [1]], (results[:, [7]]), "magenta", label="6th Degree")

plt.legend(loc="lower right")
plt.show()
