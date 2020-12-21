import csv
import numpy as np
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.set_printoptions(suppress=True)

# Global Variables #
k = 5
Age = np.array([])
Experience = np.array([])
Power = np.array([])
Salary = np.array([])
cv_errors = np.array([])
val_errors = np.array([])
# Global Variables #

with open("team_big.csv") as f:  # Check this line first if there is a problem with the execution -> team_big(2).csv #
    Total = list(csv.reader(f))

for row in Total:
    if row != Total[0]:
        Age = np.append(Age, row[4])
        Experience = np.append(Experience, row[6])
        Power = np.append(Power, row[7])
        Salary = np.append(Salary, row[8])

Ones = np.ones((len(Age), 1))  # You have warned me about my setter() function so i changed it #
Matrix = np.column_stack((Ones, Age, Experience, Power, Salary))
Matrix = Matrix.astype(np.double)


def calculate_coefficients(x, y):
    trans = np.transpose(x)
    first = np.dot(trans, x)
    second = np.linalg.inv(first)
    third = np.dot(second, trans)
    result = np.dot(third, y)
    return result


def calculate_MSE(x, y):
    result = 0.0
    for j in range(len(y)):  # There was a warning about shadowing i so i changed it as j #
        result += pow(x[j] - y[j], 2)
    result = result / len(y)
    return result


def calculate_MSE_modified(x, y):
    result = np.sum(pow(np.subtract(x, y), 2))
    result = result / len(y)
    return result


def k_fold_cv(x, y, z):  # Most of this function written using your thursday-lab-hour instructions #
    iterator = 0
    fold = round(len(x) / z)
    cv_hat = np.array([])

    for i in range(0, len(y) - fold, fold):
        Matrix_test = x[i:i + fold]
        Salary_test = y[i:i + fold]

        Matrix_train = np.delete(x, range(i, i + fold), axis=0)
        Salary_train = np.delete(y, range(i, i + fold), axis=0)

        coefficients = calculate_coefficients(Matrix_train, Salary_train)
        y_hat = np.dot(Matrix_test, coefficients)

        temp = (Salary_test - y_hat)
        Test_Sum = np.sum(temp)
        MSE_1 = Test_Sum / len(y_hat)  # Calculate First MSE value(s) #

        cv_hat = np.append(cv_hat, MSE_1)  # Append the first MSE values to the cv_hat #

        iterator += fold

    Matrix_test = x[iterator:len(x)]

    Matrix_train = np.delete(x, range(iterator, i + fold), axis=0)  # There is a weak warning here can be false #
    Salary_train = np.delete(y, range(iterator, i + fold), axis=0)

    coefficients = calculate_coefficients(Matrix_train, Salary_train)

    y_hat = np.dot(Matrix_test, coefficients)
    cv_hat = np.append(cv_hat, y_hat)

    MSE_2 = calculate_MSE(y, cv_hat)  # Calculate Second MSE value(s) #
    cv_hat = np.append(cv_hat, MSE_2)  # Append the found value to the cv_hat np.array #
    sum_oF_MSEs = np.sum(cv_hat)  # Find the total sum of the values inside the array #
    cv_error = sum_oF_MSEs / len(cv_hat)  # find cv_error #
    return cv_error


def validation(x, y):
    test_percentage = 20 / 100  # The rest is training data percentage #
    test_size = round(len(x) * test_percentage)  # Does not work without round() function #

    X_Test = x[0:test_size]
    Y_Test = y[0:test_size]

    X_Train = x[test_size:len(y)]
    Y_Train = y[test_size:len(y)]

    coefficients = calculate_coefficients(X_Train, Y_Train)
    predictions = np.dot(X_Test, coefficients)

    MSE = calculate_MSE(Y_Test, predictions)  # Calculate MSE using test data #
    return MSE  # and return the MSE of the test data #


def draw_plot(x, y, z):
    plt.figure("LAB-5 - 20150602015")
    plt.plot(range(1, z+1, 1), x, "green", label="Cross-Validation")
    plt.plot(range(1, z+1, 1), y, "red", label="Validation")
    plt.title("Cross-Validation vs Validation")  # Added title as requested #
    plt.xlabel("Shuffle Number")  # As requested axis labels are put for both x and y axis #
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")  # Render the Legend at the upper right corner #


def Main_Function(x, y):  # x is iteration amount , y is the Matrix #
    counter = 0
    cv = np.array([])
    val = np.array([])
    while counter < x:
        np.random.shuffle(y)  # thanks for the heads-up #

        shuffle_Y = y[:, (k - 1)]
        shuffle_X = y[:, 0:(k - 1)]

        shuffle_X = shuffle_X.astype(np.double)  # I guess random shuffler returns a String array ? #
        shuffle_Y = shuffle_Y.astype(np.double)  # So we need to convert them to precision-integer values #

        cv = np.append(cv, k_fold_cv(shuffle_X, shuffle_Y, k))
        val = np.append(val, validation(shuffle_X, shuffle_Y))

        counter += 1
    return cv, val


cv_errors, val_errors = Main_Function(k, Matrix)
draw_plot(cv_errors, val_errors, k)  # Graph Drawing Function #

print("\n")
print("Cross Validation Errors: ", cv_errors)
print("\n")
print("Validation Errors: ", val_errors)

plt.show()

#  This script executed flawlessly using Python 3.9 and PyCharm Community 2020.3 #

# I tried my best to bring this document as clean as I can get #
# Only a weak error exists but i could not understand  why #
# I'm not going to publish this document on my GitHub account until you submit our lab grades #
# Hopefully i don't have to get a bad grade after this #
