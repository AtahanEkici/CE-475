import csv
import random
import matplotlib.pyplot as plt
import numpy as np

Experience = np.array([])
Salary = np.array([])
Age = np.array([])
Power = np.array([])


def coef(x, y):  # First Task
    transpose = np.dot(x.T, x)
    eq1 = np.linalg.inv(transpose)
    eq2 = np.dot(eq1, x.T)
    finalize = np.dot(eq2, y)
    return finalize


def predictor(matrix, salary, coefficient):  # Second Task
    y_predictions = np.dot(matrix, coefficient)
    u_errors = abs(np.subtract(salary, y_predictions))
    return y_predictions, u_errors


def r2scores(x, y):  # Final Task
    tss = 0.0
    rss = 0.0
    x_avg = np.mean(x)
    for i in range(len(x)):
        rss += pow((x[i] - y[i]), 2)
        tss += pow(x[i] - x_avg, 2)
    return 1 - (rss / tss)


with open("team_big.csv") as f:
    CSV = list(csv.reader(f))

for row in CSV:
    if row != CSV[0]:
        Age = np.append(Age, np.double(row[4]))
        Experience = np.append(Experience, np.double(row[6]))
        Power = np.append(Power, np.double(row[7]))
        Salary = np.append(Salary, np.double(row[8]))

firstcolumn = np.ones(len(Experience))

Mat1 = np.column_stack((firstcolumn, Age, Experience, Power))

coef1 = coef(Mat1, Salary)

Y, U = predictor(Mat1, Salary, coef1)

plt.figure()
plt.scatter(Y, U, color="blue")
plt.title("Error Plot")

r2 = r2scores(Salary, Y)
print("Showing original results")
print("Original R2 Score: ", r2)

rand_array = np.array([])
z = 0
while(z < 39):
    rand_array = np.append(rand_array, random.randint(-1000, 1000))
    z = z + 1

Mat2 = np.column_stack((Mat1, rand_array))

coef2 = coef(Mat2, Salary)

Y2, U2 = predictor(Mat2, Salary, coef2)

r22 = r2scores(Salary, Y2)
print("Showing results with an added random column")
print("Modified R2 Result: ", r22)

plt.show()
