import csv
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

Country_test = np.array([])
Name_test = np.array([])
Surname_test = np.array([])
Age_test = np.array([])
Height_test = np.array([])
Experience_test = np.array([])
Power_test = np.array([])
y_test = np.array([])

Country_train = np.array([])
Name_train = np.array([])
Surname_train = np.array([])
Age_train = np.array([])
Height_train = np.array([])
Experience_train = np.array([])
Power_train = np.array([])
y_train = np.array([])
y_initial = np.array([])

with open("team_big.csv") as f:
    Total = list(csv.reader(f))

row_counter = 0

for row in Total:
    if row != Total[0]:
        y_initial = np.append(y_initial, (row[8]))
        if row_counter < 30:
            Country_train = np.append(Country_train, (row[1]))
            Name_train = np.append(Name_train, (row[2]))
            Surname_train = np.append(Surname_train, (row[3]))
            Age_train = np.append(Age_train, (row[4]))
            Height_train = np.append(Height_train, (row[5]))
            Experience_train = np.append(Experience_train, (row[6]))
            Power_train = np.append(Power_train, (row[7]))
            y_train = np.append(y_train, (row[8]))

        else:
            Country_test = np.append(Country_test, (row[1]))
            Name_test = np.append(Name_test, (row[2]))
            Surname_test = np.append(Surname_test, (row[3]))
            Age_test = np.append(Age_test, (row[4]))
            Height_test = np.append(Height_test, (row[5]))
            Experience_test = np.append(Experience_test, (row[6]))
            Power_test = np.append(Power_test, (row[7]))
            y_test = np.append(y_test, (row[8]))
        row_counter += 1

X_train = np.column_stack((Age_train, Experience_train, Power_train))
X_test = np.column_stack((Age_test, Experience_test, Power_test))

X_train = X_train.astype(float)  # To suppress the warnings that sklearn is giving #
X_test = X_test.astype(float)
y_test = y_test.astype(float)
y_train = y_train.astype(float)

regr_1 = DecisionTreeRegressor(max_depth=1, random_state=0)  # As you suggested #

Fit_Data = regr_1.fit(X_train, y_train)
Predictions = regr_1.predict(X_test)

titles = ['Age', 'Experience', 'Power']

# Decision Tree 1 #

MSE_REGR1 = mean_squared_error(y_test, Predictions)
print("------------------------------ Results For Decision Tree#1 ------------------------------")
print(MSE_REGR1)
print("The Feature Importances:", regr_1.feature_importances_)
EXText = export_text(regr_1, feature_names=titles)
print(EXText)

# Decision Tree 2 #

regr_2 = DecisionTreeRegressor(max_depth=3, random_state=0)  # As you suggested #
Fit_Data_2 = regr_2.fit(X_train, y_train)
Predictions_2 = regr_2.predict(X_test)
MSE_REGR2 = mean_squared_error(y_test, Predictions_2)
print("------------------------------ Results For Decision Tree#2 ------------------------------")
print(MSE_REGR2)
print("The Feature Importances:", regr_2.feature_importances_)
EXText_2 = export_text(regr_2, feature_names=titles)
print(EXText_2)

# Decision Tree 3 #

regr_3 = DecisionTreeRegressor(max_depth=None, random_state=0)  # As you suggested #
Fit_Data_3 = regr_3.fit(X_train, y_train)
Predictions_3 = regr_3.predict(X_test)
MSE_REGR3 = mean_squared_error(y_test, Predictions_3)
print("------------------------------ Results For Decision Tree#3 ------------------------------")
print(MSE_REGR3)
print("The Feature Importances:", regr_3.feature_importances_)
EXText_3 = export_text(regr_3, feature_names=titles)
print(EXText_3)

# Plotting Time #

plt.figure("Atahan Ekici - 20150602015")
plt.title("Decision Trees: Predictions vs. Actual Values")
plt.xlabel("Actual Salary Values For Test Data")
plt.ylabel("Salary Predictions For Test Data")

plt.plot(y_test, y_test, color='grey', label="No Error Line")
plt.scatter(y_test, Predictions, color='green', label="Decision Tree 1 (Max depth: 1)")
plt.scatter(y_test, Predictions_2, color='red', label="Decision Tree 2 (Max depth: 3)")
plt.scatter(y_test, Predictions_3, color='blue', label="Decision Tree 3 (Max depth: None)")
plt.legend(loc="upper left")
plt.show()
