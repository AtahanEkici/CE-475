import csv
import matplotlib.pyplot as plt
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def setter(x, y):
    counter = 0
    while counter < len(x):
        x = np.append(x, y)
        counter += 1
    return x


def question1(x, y):
    trans = np.transpose(x)
    first = np.dot(trans, x)
    second = np.linalg.inv(first)
    third = np.dot(second, trans)
    result = np.dot(third, y)
    return result


def question1_modified(x, y):
    counter = 0
    result = np.array([])

    while counter < len(x):
        trans = np.transpose(x[counter])
        first = np.dot(trans, x[counter])
        second = np.linalg.inv(first)
        third = np.dot(second, trans)
        result = np.dot(third, y[counter])
        counter += 1
    return result


def question2(x, y, z):
    Y = np.dot(x, z)
    U = abs(np.subtract(y, Y))
    plt.figure()
    plt.scatter(Y, U, color="blue")
    plt.title("Original")
    plt.legend("Without CV")
    return Y


def question2_modified(x, y, z):
    counter = 0
    while counter < len(x):
        Y = np.dot(x[counter], z)
        counter += 1
    U = abs(np.subtract(y[counter], Y))

    plt.figure()
    plt.scatter(Y, U, color="blue")
    plt.title("Cross Validation")
    return Y,U


def question3(x, y):
    counter = 0
    rss = 0.0
    tss = 0.0
    avg_x = np.mean(x)
    while counter < len(x):
        rss += pow(x[counter] - y[counter], 2)
        tss += pow(x[counter] - avg_x, 2)
        counter += 1
    r2 = 1 - (rss / tss)
    return r2


def array_splitter(x, y):  # Since the matrix is in 39,4 format we need to divide it by 4 to be able to get 10
    # Last fold's size will be 3 since 39 % 4 is not 0
    a = np.array_split(x, y)
    return a


def calculate_MSE(x, y):
    counter = 0
    result = 0.0
    while counter < len(x):
        result += pow(x[counter] - y[counter], 2)
        counter += 1
    result = result / len(y)
    return result


def new_func(x):
    counter = 0
    result = 0.0
    while counter < len(x):
        result += x[counter]
        counter += 1
    return result


def convert_float(x):
    counter = 0
    while counter < len(x):
        float(x[counter])
        counter += 1


Matrix = np.array([])

Age = np.array([])
Experience = np.array([])
Power = np.array([])
Salary = np.array([])

with open("team_big.csv") as f:
    All = list(csv.reader(f))

for row in All:
    if row != All[0]:
        Age = np.append(Age, np.float(row[4]))  # Age
        Experience = np.append(Experience, np.float(row[6]))  # Experience
        Power = np.append(Power, np.float(row[7]))  # Power
        Salary = np.append(Salary, np.float(row[8]))  # Salary

temp = setter(Matrix, 1)

Matrix = np.column_stack((temp, Age))
Matrix = np.column_stack((Matrix, Experience))
Matrix = np.column_stack((Matrix, Power))

q1_values = np.array(array_splitter(Matrix, 10))
sal_values = np.array(array_splitter(Salary, 10))


def lab4_q1(x, y):
    counter = 0

    temp_sal = y
    temp_sal = np.delete(temp_sal, len(y)-1)

    temp_mat = x
    temp_mat = np.delete(temp_mat, len(x)-1)

    while counter < len(temp_mat) - 1:
        test_data = np.array(temp_mat[counter])

        train_data = np.array(temp_mat)
        train_data = np.delete(train_data, counter)

        sal_train = np.array(temp_sal)
        sal_train = np.array(np.delete(sal_train, counter))

        train_coef = question1(train_data[counter], sal_train[counter])  # Regression coefficients #
        cv_hat = np.dot(test_data, train_coef)  # cv_hat calculation #
        counter += 1
    return cv_hat


temp_x = q1_values[9]
temp_y = sal_values[9]

B_1 = question1(Matrix, Salary)
y_hat = question2(Matrix, Salary, B_1)
original_MSE = calculate_MSE(y_hat, Salary)
print("MSE without cross-validation:", original_MSE)


cv_hat = lab4_q1(q1_values, sal_values)
mse = calculate_MSE(cv_hat, sal_values)
new_mse = new_func(mse)
print("MSE with cross validation:", new_mse)

#plt.show()
