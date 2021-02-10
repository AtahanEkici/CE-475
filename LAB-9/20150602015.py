import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# CSV File Read Dump #
ACE_1 = np.array([])
ACE_2 = np.array([])
Result = np.array([])

ACE_1_Train = np.array([])
ACE_2_Train = np.array([])
Result_Train = np.array([])

ACE_1_Test = np.array([])
ACE_2_Test = np.array([])
Result_Test = np.array([])

Matrix_Train = np.array([])
Matrix_Test = np.array([])
Matrix_Test_2 = np.array([])
# CSV File Read Dump #

# Train and Test 2 Values #
ACE_1_Test_2 = np.array([])
ACE_2_Test_2 = np.array([])
Result_Test_2 = np.array([])
# Train and Test 2 Values #

# Stored Indices #
Zeros_Linear = np.array([])
Ones_Linear = np.array([])

Zeros_Poly = np.array([])
Ones_Poly = np.array([])

Zeros_RBF = np.array([])
Ones_RBF = np.array([])
# Stored Indices #

with open("Grand-slams-men-2013.csv") as f:
    Total = list(csv.reader(f))

row_counter = 0

for row in Total:
    if row != Total[0]:
        if row_counter < 100:  # First 100 values is training data #
            ACE_1_Train = np.append(ACE_1_Train, np.double(row[10]))
            ACE_2_Train = np.append(ACE_2_Train, np.double(row[28]))
            Result_Train = np.append(Result_Train, np.double(row[3]))

        elif 100 <= row_counter < 200:  # The other part is the test data as requested #
            ACE_1_Test = np.append(ACE_1_Test, np.double(row[10]))
            ACE_2_Test = np.append(ACE_2_Test, np.double(row[28]))
            Result_Test = np.append(Result_Test, np.double(row[3]))

        else:
            ACE_1_Test_2 = np.append(ACE_1_Test_2, np.double(row[10]))
            ACE_2_Test_2 = np.append(ACE_2_Test_2, np.double(row[28]))
            Result_Test_2 = np.append(Result_Test_2, np.double(row[3]))
        row_counter += 1

Matrix_Train = np.column_stack((ACE_1_Train, ACE_2_Train))
Matrix_Test = np.column_stack((ACE_1_Test, ACE_2_Test))
Matrix_Test_2 = np.column_stack((ACE_1_Test_2, ACE_2_Test_2))

Matrix_ACE1_Test = np.array([])
Matrix_ACE2_Test = np.array([])

Matrix_ACE1_Test = np.append(Matrix_ACE1_Test, ACE_1_Test)
Matrix_ACE1_Test = np.append(Matrix_ACE1_Test, ACE_1_Test_2)

Matrix_ACE2_Test = np.append(Matrix_ACE2_Test, ACE_2_Test)
Matrix_ACE2_Test = np.append(Matrix_ACE2_Test, ACE_2_Test_2)

print(Matrix_Train.shape, Matrix_Test.shape)
print(Matrix_Test_2.shape, Result_Test_2.shape)
print(Result_Train.shape, Result_Test.shape)
print(Matrix_ACE1_Test.shape, Matrix_ACE2_Test.shape)

Linear_predictions_All = np.array([])
Poly_predictions_All = np.array([])
RBF_predictions_All = np.array([])

# ---------------- Linear Model ---------------- #
SVC_Linear = SVC(kernel='linear')
SVC_Linear.fit(Matrix_Train, Result_Train)
Linear_predictions = SVC_Linear.predict(Matrix_Test)
Linear_predictions_2 = SVC_Linear.predict(Matrix_Test_2)
Linear_predictions_All = np.append(Linear_predictions_All, Linear_predictions)
Linear_predictions_All = np.append(Linear_predictions_All, Linear_predictions_2)
# ---------------- Linear Model ---------------- #

# ---------------- Poly Model ---------------- #
SVC_Poly = SVC(kernel='poly')
SVC_Poly.fit(Matrix_Train, Result_Train)
Poly_predictions = SVC_Poly.predict(Matrix_Test)
Poly_predictions_2 = SVC_Poly.predict(Matrix_Test_2)
Poly_predictions_All = np.append(Poly_predictions_All, Poly_predictions)
Poly_predictions_All = np.append(Poly_predictions_All, Poly_predictions_2)
# ---------------- Poly Model ---------------- #

# ---------------- RBF Model ---------------- #
SVC_RBF = SVC(kernel='rbf', gamma='auto')
SVC_RBF.fit(Matrix_Train, Result_Train)
RBF_predictions = SVC_RBF.predict(Matrix_Test)
RBF_predictions_2 = SVC_RBF.predict(Matrix_Test_2)
RBF_predictions_All = np.append(RBF_predictions_All, RBF_predictions)
RBF_predictions_All = np.append(RBF_predictions_All, RBF_predictions_2)
# ---------------- RBF Model ---------------- #

# print(Linear_predictions)
# print(Poly_predictions)
# print(Poly_predictions)

# ------------ Linear Indices ------------ #
Ones_Linear = np.where(Linear_predictions_All == 1)
Zeros_Linear = np.where(Linear_predictions_All == 0)
# print(Ones_Linear, "\n", Zeros_Linear)
# ------------ Linear Indices ------------ #

# ------------ Poly Indices ------------ #
Ones_Poly = np.where(Poly_predictions_All == 1)
Zeros_Poly = np.where(Poly_predictions_All == 0)
# print(Ones_Poly, "\n", Zeros_Poly)
# ------------ Poly Indices ------------ #

# ------------ RBF Indices ------------ #
Ones_RBF = np.where(RBF_predictions_All == 1)
Zeros_RBF = np.where(RBF_predictions_All == 0)
# print(Ones_RBF, "\n", Zeros_RBF)
# ------------ RBF Indices ------------ #

# ------------ Plotting Segment ------------ #
figure1 = plt.figure("20150602015 - Linear")  # Figure 1 #
plt.title('SVC with linear kernel')
plt.xlabel('Number Of Aces by Player_1')
plt.ylabel('Number Of Aces by Player_2')
plt.scatter(Matrix_ACE1_Test[Zeros_Linear], Matrix_ACE2_Test[Zeros_Linear], c=["green"],
            label='Prediction: 1 st Player Won')
plt.scatter(Matrix_ACE1_Test[Ones_Linear], Matrix_ACE2_Test[Ones_Linear], c=["red"], label='Prediction: 2nd Player Won')
plt.legend(loc='lower right')

figure2 = plt.figure("20150602015 - Polynomial")  # Figure 2 #
plt.title('SVC with polynomial kernel')
plt.xlabel('Number Of Aces by Player_1')
plt.ylabel('Number Of Aces by Player_2')
plt.scatter(Matrix_ACE1_Test[Zeros_Poly], Matrix_ACE2_Test[Zeros_Poly], c=["green"],
            label='Prediction: 1 st Player Won')
plt.scatter(Matrix_ACE1_Test[Ones_Poly], Matrix_ACE2_Test[Ones_Poly], c=["red"], label='Prediction: 2nd Player Won')
plt.legend(loc='lower right')

figure3 = plt.figure("20150602015 - Radial")  # Figure 3 #
plt.title('SVC with radial kernel')
plt.xlabel('Number Of Aces by Player_1')
plt.ylabel('Number Of Aces by Player_2')
plt.scatter(Matrix_ACE1_Test[Zeros_RBF], Matrix_ACE2_Test[Zeros_RBF], c=["green"], label='Prediction: 1 st Player Won')
plt.scatter(Matrix_ACE1_Test[Ones_RBF], Matrix_ACE2_Test[Ones_RBF], c=["red"], label='Prediction: 2nd Player Won')
plt.legend(loc='lower right')

plt.show()
# ------------ Plotting Segment ------------ #
