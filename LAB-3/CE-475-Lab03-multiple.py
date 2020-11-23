import csv
import numpy as np
import matplotlib.pyplot as plt

Ones = np.array([])
Age = np.array([])  # X1 Vector for analysis
Experience = np.array([])  # X2 Vector for analysis
Power = np.array([])  # X3 Vector for analysis
yerror = np.array([])
y_pred = np.array([])
x_rand = np.array([])
EXmat = np.array([])

Salary = np.array([])  # Y Vector for analysis

with open("team_big.csv") as f:
    lst = list(csv.reader(f))  # - Read csv file

for row in lst:
    if row != lst[0]:  # Skip Column headers
        Age = np.append(Age, int(row[4]))
        Ones = np.append(Ones, 1)  # - First column vector are all 1s
        Experience = np.append(Experience, int(row[6]))  # - experience vector into array
        Power = np.append(Power, float(row[7]))  # - experience vector into array
        Salary = np.append(Salary, int(row[8]))  # - salary vector into array

dimX = len(Ones)

# --- Extracting & Arranging input matrix(20 Pts)---
# --------------------------------------------------     
print("CE-475 LAB-01 MULTIPLE LINEAR REGRESSION -TEAM_BIG DATA \n")
Xmat = np.column_stack((Ones, Age, Experience, Power))  # - Stack extracted input vectors in Xmat
# Xmat = np.mat(Xmat)
print("----Input Matrix Xmat[", dimX, ",4]----")
print(Xmat)
print("---------------")

# --- Calculation of coefficients(30 Pts)---
# ------------------------------------------ 
# -- PLEASE NOTE numpy.mat(array[]) assigment used so that *, **. etc. OPERATORS ARE OVERLOADED 
# --- for matrix objects to allow matrix operations be written as ORDINAEY SCALAR OPERATIONS
# ------------------------------------------    
Xmat_t = np.transpose(Xmat)  # - Calculate the Xmat tranpose matrix
print("-Input Matrix Transposed Xmat_t[4,", dimX, "]-")
print(Xmat_t)
print("---------------")
Xmat_pr = np.dot(Xmat_t, Xmat)  # - Calculates the product transpose(Xmat) and Xmat
print("----Input Matrix multiplied by transpose Transposed Xmat_pr[4,4]----")
print(Xmat_pr)
print("---------------")
Xmat_prI = np.linalg.inv(Xmat_pr)  # Takes the Inverse of product matrix
print("----Inverse of product Xmat_pr_inv[4,4]----")
print(Xmat_prI)  # Takes the Inverse of product matrix
print("---------------")
X_last = np.dot(Xmat_prI, Xmat_t)  # Calculates the product (X_last) of Inverse matrix and transpose matrix
print("-Last Matrix for coefficients X_last[", dimX, ",1]-")
print(X_last)
print("---------------")
y = np.transpose(Salary)  # - We take the transpose of response vector for coefficient calculation
print("-Response Vector [1,", dimX, "]-")
print(y)  # - we prefer horizontal display for easy checking on console screen
print("---------------")
Coefs = np.dot(X_last, Salary)  # - Calculation of coefficients by matrix multiplication
b = np.transpose(Coefs)  # - We take the transpose for horizontal display
print("Coefficients Vector Coefs[4,,1]")
print(b)
print("---------------")
y_pred = np.dot(Xmat, Coefs)
print("-Predicted Response Vector with coefficients [1,", dimX, "]-")
print(np.transpose(y_pred))  # - we prefer horizontal display for easy checking on console screen
print("---------------")
# --------- Plots (25 Pts)-----------
# -----------------------------------
yerror = abs(np.subtract(Salary, y_pred))
print("-Error Vector as calculated y_err[1,", dimX, "]-")
print(yerror.T)  # - we prefer horizontal display for easy checking on console screen
print("---------------")
# --Plotting---
plt.figure()
plt.xlabel("Predicted y", color="red", size="15")
plt.ylabel("Abs. Error", color="red", size="15")
plt.scatter(yerror, y_pred, color="blue")


# --------- R**2 calculation function definition (25 Pts)-----------
# -----------------------------------
def GetR2(u, w):  # Defined as a function for repeatability
    j = 0
    rss = 0.0
    tss = 0.0
    avg = np.mean(u)
    while j < len(u):
        rss += (u[j] - w[j]) ** 2
        tss += (u[j] - avg) ** 2
        j += 1
    Res = 1 - (rss / tss)
    return Res


R2 = GetR2(y, y_pred)  # R**2 calculation
print("-----------\n", "Calculated R**2 = ", R2, "\n------------")
# -----------------------------
# Adding an extra random column
# -----------------------------
x_rand = np.random.randint(-1000, 1000, dimX)
EXmat = np.column_stack((Xmat, x_rand))
# - Repeat matrix calculations
Xmat_t = np.transpose(EXmat)
Xmat_pr = np.dot(Xmat_t, EXmat)
Xmat_prI = np.linalg.inv(Xmat_pr)
X_last = np.dot(Xmat_prI, Xmat_t)
Coefs = np.dot(X_last, Salary)
y_pred = np.dot(EXmat, Coefs)
# Recalculate R**2
R2 = GetR2(y, y_pred)
print("-----------\n", "Re-Calculated R**2 with extra vector = ", R2, "\n------------")
# Display plots
plt.show()
