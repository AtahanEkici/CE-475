import csv
import numpy as np
import random

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.set_printoptions(suppress=True)

Age = np.array([])
Experience = np.array([])
Power = np.array([])
Salary = np.array([])
Model_Values = np.array([])


def square(sqr):
    result = (sqr * sqr)
    return result


def avg(c):
    c = np.asarray(c)
    counter = 0
    summary = 0.0
    while counter < len(c):
        summary += c[counter]
        counter += 1
    average = summary / len(c)
    return average


def adjusted_R2_Scores(x, y, d):
    n = len(y)
    counter = 0
    rss = 0.0
    tss = 0.0
    avg_x = avg(x)
    while counter < len(x):
        rss += square(x[counter] - y[counter])
        tss += square(x[counter] - avg_x)
        counter += 1
    adjusted_r2 = 1 - ((rss / (n - d - 1)) / (tss / (n - 1)))
    return adjusted_r2


def calculate_coefficients(x, y):
    trans = np.transpose(x)
    first = np.dot(trans, x)
    second = np.linalg.inv(first)
    third = np.dot(second, trans)
    result = np.dot(third, y)
    return result


def calculate_predictions(x, y):
    Y = np.dot(x, y)
    return Y


def R2_Scores(x, y):
    counter = 0
    rss = 0.0
    tss = 0.0
    avg_x = avg(x)
    while counter < len(y):
        rss += square(x[counter] - y[counter])
        tss += square(x[counter] - avg_x)
        counter += 1
    r2 = 1 - (rss / tss)
    return r2


def append_random(x):
    counter = 0
    temp1 = np.array([])
    while counter < x:
        temp1 = np.append(temp1, random.randint(-1000, 1000))
        counter += 1
    temp1 = temp1.astype(np.double)
    return temp1


with open("team_big.csv") as f:  # Check this line first if there is a problem with the execution -> team_big(3).csv #
    Total = list(csv.reader(f))

for row in Total:
    if row != Total[0]:
        Age = np.append(Age, np.double(row[4]))
        Experience = np.append(Experience, np.double(row[6]))
        Power = np.append(Power, np.double(row[7]))
        Salary = np.append(Salary, np.double(row[8]))

Ones = np.ones((len(Age), 1))  # initialize Ones array #
Randoms = append_random(len(Age))  # initialize random array #
Matrix = np.column_stack((Ones, Age, Experience, Power, Randoms))  # stack all #

B_1 = calculate_coefficients(Matrix, Salary)
Y_1 = calculate_predictions(Matrix, B_1)
R2_1 = R2_Scores(Salary, Y_1)
R2_2 = adjusted_R2_Scores(Salary, Y_1, 0)
# print(R2_1, R2_2) To check if the R2 scores are equal or not #
Model_Values = np.append(Model_Values, R2_1)  # Append M4 #


def calculate_Model(x, y):
    Matrix_temp = x  # since we are going to change the original might as well store it #
    size = y

    temp_R2 = np.array([])
    result_Array = np.array([])

    deletions = np.array([])
    deletion_index = np.array([])

    for i in range(0, y, 1):
        size = size - 1  # since we remove an entire column the size shrinks obviously #
        for j in range(0, size, 1):
            temp_column = np.delete(Matrix_temp, j, 1)  # in each iteration delete one of them but note the ones #
            B = calculate_coefficients(temp_column, Salary)
            Y = calculate_predictions(temp_column, B)
            R2 = R2_Scores(Salary, Y)
            temp_R2 = np.append(temp_R2, R2)  # iterate the current matrix's R2 values one by one #
            # print(size, len(temp_R2))

            if len(temp_R2) == size:
                max_R2_index = np.argmax(temp_R2)

                deletions = np.vstack(Matrix_temp[:, [max_R2_index]])
                # print(deletions, "\n")

                Matrix_temp = np.delete(Matrix_temp, max_R2_index, 1)
                # print("deletion")
                # print(Matrix_temp, "\n")
                deletion_index = np.append(deletion_index, max_R2_index)

                # print(temp_R2, min_R2, min_R2_index)

                B_2 = calculate_coefficients(Matrix_temp, Salary)
                Y_2 = calculate_predictions(Matrix_temp, B_2)
                R2_D2 = adjusted_R2_Scores(Salary, Y_2, len(result_Array))  # Calculate adjusted R2 Values #
                result_Array = np.append(result_Array, R2_D2)
                temp_R2 = np.empty(0)  # delete the temps #
    #deletions = np.reshape(deletions, (len(Age), -1))
    return result_Array, deletions, deletion_index


Results, Deletions, Indices = calculate_Model(Matrix, 5)
Model_Values = np.append(Model_Values, Results)

print("      M4         ", "M3        ", "M2        ", "M1        ", "M0        ")
print(Model_Values)  # M0 gives negative values  #
# print(Deletions)
# print(Indices)

# i could not figure out why the M0 is getting negative results #
