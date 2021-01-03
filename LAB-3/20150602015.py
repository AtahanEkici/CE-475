import csv
import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(suppress=True)  # This is for getting rid of annoying scientific e notations


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


def setter(x, y):
    counter = 0
    while counter < 39:
        x = np.append(x, y)
        counter += 1
    return x


def setter2(x):
    counter = 0
    while counter < 39:
        x = np.append(x, random.randint(-1000, 1000))
        counter += 1
    return x


def question1(x, y):
    trans = np.transpose(x)
    first = np.dot(trans, x)
    second = np.linalg.inv(first)
    third = np.dot(second, trans)
    result = np.dot(third, y)
    return result


def question2(x, y, z):
    Y = np.dot(x, z)
    U = abs(np.subtract(y, Y))
    plt.figure()
    plt.scatter(Y, U, color="blue")
    plt.title("Scattered Error Plot")
    return Y


Matrix = np.array([])
temp = np.array([])
temp2 = np.array([])

Age = np.array([])
Experience = np.array([])
Power = np.array([])
Salary = np.array([])

with open("team_big.csv") as f:
    All = list(csv.reader(f))

for row in All:
    if row != All[0]:
        Age = np.append(Age, np.double(row[4]))  # Age
        Experience = np.append(Experience, np.double(row[6]))  # Experience
        Power = np.append(Power, np.double(row[7]))  # Power
        Salary = np.append(Salary, np.double(row[8]))  # Salary

temp = setter(Matrix, 1)

Matrix = np.column_stack((temp, Age))
Matrix = np.column_stack((Matrix, Experience))
Matrix = np.column_stack((Matrix, Power))

print(Matrix.shape, Salary.shape)

B = question1(Matrix, Salary)
print("\n")
print(B)

Y = question2(Matrix, Salary, B)   # Predicted Salary values


def question3(x, y):
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


R2 = question3(Salary, Y)
print("Showing original results:")
print("R2 Results: ", R2)

print("Showing results with an added random column:")
Matrix = np.column_stack((Matrix, setter2(temp2)))
B_2 = question1(Matrix, Salary)
Y2 = question2(Matrix, Salary, B_2)
R2_2 = question3(Salary, Y2)
print("R2 Results: ", R2_2)


plt.show()  # shows both the original and modified first one is the original other is the modified version
