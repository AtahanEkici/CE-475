import csv
import numpy as np
# import pandas as pd # This library is used for exporting the selected prediction values #
import matplotlib.pyplot as plt

# Sklearn Imports #
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
# Sklearn Imports #

# Other libraries #
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor
# Other libraries #


np.set_printoptions(suppress=True)  # This is for getting rid of scientific e notations #
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  # Ignore Deprecation Warnings #
# ------------------ Array References ------------------ #
x1 = np.array([])
x2 = np.array([])
x3 = np.array([])
x4 = np.array([])
x5 = np.array([])
x6 = np.array([])

x1_2 = np.array([])
x2_2 = np.array([])
x3_2 = np.array([])
x4_2 = np.array([])
x5_2 = np.array([])
x6_2 = np.array([])

Y = np.array([])
# ------------------ Array References ------------------ #

# ------------------ List References ------------------ #
scores = []  # Accuracy Scores of the models #
names = []  # Names of the models to be able to fit through a bar graph #

k_fold_cv_avg = []  # average of k-fold cross validation score #
k_fold_n_mse_avg = []  # average of negative mean squared error rate #
# ------------------ List References ------------------ #

with open("Data.csv") as f:
    Data = list(csv.reader(f))
counter = 0
for row in Data:
    if row != Data[0]:
        if counter < 100:
            x1 = np.append(x1, np.double(row[1]))
            x2 = np.append(x2, np.double(row[2]))
            x3 = np.append(x3, np.double(row[3]))
            x4 = np.append(x4, np.double(row[4]))
            x5 = np.append(x5, np.double(row[5]))
            x6 = np.append(x6, np.double(row[6]))
            Y = np.append(Y, np.double(row[7]))

        else:
            x1_2 = np.append(x1_2, np.double(row[1]))
            x2_2 = np.append(x2_2, np.double(row[2]))
            x3_2 = np.append(x3_2, np.double(row[3]))
            x4_2 = np.append(x4_2, np.double(row[4]))
            x5_2 = np.append(x5_2, np.double(row[5]))
            x6_2 = np.append(x6_2, np.double(row[6]))
        counter += 1

X_Train = np.column_stack((x1, x2, x3, x4, x5, x6))
X_Test = np.column_stack((x1_2, x2_2, x3_2, x4_2, x5_2, x6_2))

folds = KFold(n_splits=5, random_state=None)

# ---------------------------------------- Linear Model ---------------------------------------- #
Linear_Regression = LinearRegression()
Linear_Regression.fit(X_Train, Y)
Linear_Score = Linear_Regression.score(X_Train, Y)
Linear_Predictions = Linear_Regression.predict(X_Test)

print("Linear Model Predictions:\n", Linear_Predictions)
print("\nLinear Model Score:", Linear_Score)

Linear_cv_scores = cross_val_score(Linear_Regression, X_Train, Y, scoring='r2', cv=folds)
print("Linear K-Fold CV Scores:", Linear_cv_scores)

Linear_cv_NMSE_scores = cross_val_score(Linear_Regression, X_Train, Y, scoring='neg_mean_squared_error', cv=folds)
print("Linear K-Fold MSE Scores:", Linear_cv_NMSE_scores)

print("Linear K-Fold CV Score Average: ", np.average(Linear_cv_scores))
k_fold_cv_avg.append(np.average(Linear_cv_scores))

print("Linear K-Fold CV N-MSE Score Average: ", np.average(Linear_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Linear_cv_NMSE_scores))

scores.append(Linear_Score)
names.append('Linear')
# ---------------------------------------- Linear Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Polynomial Model ---------------------------------------- #
Polynomial_Regression = make_pipeline(PolynomialFeatures(), LinearRegression())
Polynomial_Regression.fit(X_Train, Y)
Polynomial_Score = Polynomial_Regression.score(X_Train, Y)
Polynomial_Predictions = Polynomial_Regression.predict(X_Test)

print("Polynomial Model Predictions:\n", Polynomial_Predictions)
print("\nPolynomial Model Score:", Polynomial_Score)

Polynomial_cv_scores = cross_val_score(Polynomial_Regression, X_Train, Y, scoring='r2', cv=folds)
print("Polynomial K-Fold CV Scores:", Polynomial_cv_scores)

Polynomial_cv_NMSE_scores = cross_val_score(Polynomial_Regression, X_Train, Y, scoring='neg_mean_squared_error',
                                            cv=folds)
print("Polynomial K-Fold MSE Scores:", Polynomial_cv_NMSE_scores)

print("Polynomial K-Fold CV N-MSE Score Average: ", np.average(Polynomial_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Polynomial_cv_NMSE_scores))

print("Polynomial CV Score Average: ", np.average(Polynomial_cv_scores))
k_fold_cv_avg.append(np.average(Polynomial_cv_scores))

scores.append(Polynomial_Score)
names.append('Poly.')
# ---------------------------------------- Polynomial Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Random Forest Model ---------------------------------------- #
Random_Forest = RandomForestRegressor(random_state=0, min_samples_split=5, max_samples=80)
Random_Forest.fit(X_Train, Y)

Random_Forest_Score = Random_Forest.score(X_Train, Y)
Random_Forest_Predictions = Random_Forest.predict(X_Test)

Random_Forest_cv_scores = cross_val_score(Random_Forest, X_Train, Y, scoring='r2', cv=folds)
print("Random Forest K-Fold CV Scores:", Random_Forest_cv_scores)

Random_Forest_cv_NMSE_scores = cross_val_score(Random_Forest, X_Train, Y, scoring='neg_mean_squared_error', cv=folds)
print("Random Forest K-Fold N-MSE Scores:", Random_Forest_cv_NMSE_scores)

print("Random Forest K-Fold CV Score Average: ", np.average(Random_Forest_cv_scores))
k_fold_cv_avg.append(np.average(Random_Forest_cv_scores))

print("Random Forest K-Fold CV N-MSE Score Average: ", np.average(Random_Forest_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Random_Forest_cv_NMSE_scores))

print("Random Forest Model Predictions:\n", Random_Forest_Predictions)
print("\nRandom Forest Model Score:", Random_Forest_Score)

scores.append(Random_Forest_Score)
names.append('R.Forest')
# ---------------------------------------- Random Forest Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Decision Tree Model ---------------------------------------- #
Decision_Tree = DecisionTreeRegressor(random_state=0, max_depth=5)
Decision_Tree.fit(X_Train, Y)

Decision_Tree_Score = Decision_Tree.score(X_Train, Y)
Decision_Tree_Predictions = Decision_Tree.predict(X_Test)

Decision_Tree_cv_R2_scores = cross_val_score(Decision_Tree, X_Train, Y, scoring='r2', cv=folds)
print("Decision Tree  K-Fold CV Scores:", Decision_Tree_cv_R2_scores)

Decision_Tree_cv_NMSE_scores = cross_val_score(Decision_Tree, X_Train, Y, scoring='neg_mean_squared_error',
                                               cv=folds)
print("Decision Tree  K-Fold MSE Scores:", Decision_Tree_cv_NMSE_scores)

print("Decision Tree Model CV Score Average: ", np.average(Decision_Tree_cv_R2_scores))
k_fold_cv_avg.append(np.average(Decision_Tree_cv_R2_scores))

print("Decision Tree Model N-MSE Score Average: ", np.average(Decision_Tree_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Decision_Tree_cv_NMSE_scores))

print("Decision Tree Model Predictions:\n", Decision_Tree_Predictions)
print("\nDecision Tree Model Score:", Decision_Tree_Score)

scores.append(Decision_Tree_Score)
names.append('D.Tree')
# ---------------------------------------- Decision Tree Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Logistic Regression Model ---------------------------------------- #
Logistic_Regression = make_pipeline(StandardScaler(), LogisticRegression(random_state=None))
Logistic_Regression.fit(X_Train, Y)

Logistic_Regression_Score = Logistic_Regression.score(X_Train, Y)
Logistic_Regression_Predictions = Logistic_Regression.predict(X_Test)

Logistic_Regression_cv_scores = cross_val_score(Logistic_Regression, X_Train, Y, scoring='r2', cv=folds)
print("Logistic Regression K-Fold CV Scores:", Logistic_Regression_cv_scores)

Logistic_Regression_cv_NMSE_scores = cross_val_score(Logistic_Regression, X_Train, Y,
                                                     scoring='neg_mean_squared_error', cv=folds)
print("Logistic Regression K-Fold MSE Scores:", Logistic_Regression_cv_NMSE_scores)

print("Logistic RegressionModel CV Score Average: ", np.average(Logistic_Regression_cv_scores))
k_fold_cv_avg.append(np.average(Logistic_Regression_cv_scores))

print("Logistic RegressionModel N-MSE Score Average: ", np.average(Decision_Tree_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Decision_Tree_cv_NMSE_scores))

print("Logistic Regression Model Predictions:\n", Logistic_Regression_Predictions)
print("\nLogistic Regression Model Score:", Logistic_Regression_Score)

scores.append(Logistic_Regression_Score)
names.append('Logistic')
# ---------------------------------------- Logistic Regression Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Gradient Booster Model ---------------------------------------- #
learning_rates = [0.10, 0.25, 0.50, 0.75, 1.00]

k_fold_cv_avg_gb = []
k_fold_n_mse_avg_gb = []
Gradient_Boosting_Score = 0

for i in learning_rates:
    Gradient_Boosting = GradientBoostingRegressor(learning_rate=i, random_state=0)
    Gradient_Boosting.fit(X_Train, Y)

    Gradient_Boosting_Score = Gradient_Boosting.score(X_Train, Y)
    Gradient_Boosting_Predictions = Gradient_Boosting.predict(X_Test)

    Gradient_Boosting_cv_scores = cross_val_score(Gradient_Boosting, X_Train, Y, scoring='r2', cv=folds)
    print("Gradient Boosting K-Fold CV Scores:", Gradient_Boosting_cv_scores)

    Gradient_Boosting_cv_NMSE_scores = cross_val_score(Gradient_Boosting, X_Train, Y, scoring='neg_mean_squared_error',
                                                       cv=folds)
    print("Gradient Boosting K-Fold MSE Scores:", Gradient_Boosting_cv_NMSE_scores)

    print("Gradient Boosting Model CV Score Average: ", np.average(Gradient_Boosting_cv_scores))
    k_fold_cv_avg_gb.append(np.average(Gradient_Boosting_cv_scores))

    print("Gradient Boosting Model N-MSE Score Average: ", np.average(Gradient_Boosting_cv_NMSE_scores))
    k_fold_n_mse_avg_gb.append(np.average(Gradient_Boosting_cv_NMSE_scores))

    print("Learning rate: ", i)
    print("Gradient Boosting Model Predictions :\n", Gradient_Boosting_Predictions)
    print("\nGradient Boosting Model Score:", Gradient_Boosting_Score)
    print("\n")

scores.append(Gradient_Boosting_Score)
names.append('G.Boost')

k_fold_n_mse_avg.append(max(k_fold_n_mse_avg_gb))
k_fold_cv_avg.append(max(k_fold_cv_avg_gb))
# ---------------------------------------- Gradient Booster Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- KNN Model ---------------------------------------- #
KNN_Regression = KNeighborsRegressor(n_neighbors=6, leaf_size=150)
KNN_Regression.fit(X_Train, Y)

KNN_Regression_Score = KNN_Regression.score(X_Train, Y)
KNN_Regression_Predictions = KNN_Regression.predict(X_Test)

KNN_Regression_cv_scores = cross_val_score(KNN_Regression, X_Train, Y, scoring='r2', cv=folds)
print("KNN Regression K-Fold CV Scores:", KNN_Regression_cv_scores)

KNN_Regression_cv_NMSE_scores = cross_val_score(KNN_Regression, X_Train, Y, scoring='neg_mean_squared_error',
                                                cv=folds)
print("KNN Regression K-Fold MSE Scores:", KNN_Regression_cv_NMSE_scores)

print("KNN Regression Model CV Score Average: ", np.average(KNN_Regression_cv_scores))
k_fold_cv_avg.append(np.average(KNN_Regression_cv_scores))

print("KNN Regression Model N-MSE Score Average: ", np.average(KNN_Regression_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(KNN_Regression_cv_NMSE_scores))

print("K Nearest Neighbour Model Predictions :\n", KNN_Regression_Predictions)
print("\nK Nearest Neighbour Model Score:", KNN_Regression_Score)
scores.append(KNN_Regression_Score)
names.append('KNN')
# ---------------------------------------- KNN Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- XGB Regression Model ---------------------------------------- #
XGB_Regression = XGBRegressor(random_state=0)
XGB_Regression.fit(X_Train, Y)

XGB_Regression_Score = XGB_Regression.score(X_Train, Y)
XGB_Regression_Predictions = XGB_Regression.predict(X_Test)

XGB_Regression_cv_scores = cross_val_score(XGB_Regression, X_Train, Y, scoring='r2', cv=folds)
print("XGB Regression K-Fold CV Scores:", XGB_Regression_cv_scores)

XGB_Regression_cv_NMSE_scores = cross_val_score(XGB_Regression, X_Train, Y, scoring='neg_mean_squared_error',
                                                cv=folds)
print("XGB Regression K-Fold MSE Scores:", XGB_Regression_cv_NMSE_scores)

print("XGB Regression Model CV Score Average: ", np.average(XGB_Regression_cv_scores))
k_fold_cv_avg.append(np.average(XGB_Regression_cv_scores))

print("XGB Regression Model N-MSE Score Average: ", np.average(XGB_Regression_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(XGB_Regression_cv_NMSE_scores))

print("XGB Regression Model Predictions :\n", XGB_Regression_Predictions)
print("\nXGB Regression Model Score:", XGB_Regression_Score)

scores.append(XGB_Regression_Score)
names.append('XGB')
# ---------------------------------------- XGB Regression Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- LightGBM Regression Model ---------------------------------------- #
LGBM_Regression = LGBMRegressor(random_state=0)
LGBM_Regression.fit(X_Train, Y)

LGBM_Regression_Score = LGBM_Regression.score(X_Train, Y)
LGBM_Regression_Predictions = LGBM_Regression.predict(X_Test)

LGBM_Regression_cv_scores = cross_val_score(LGBM_Regression, X_Train, Y, scoring='r2', cv=folds)
print("Light GBM Regression K-Fold CV Scores:", LGBM_Regression_cv_scores)

LGBM_Regression_cv_NMSE_scores = cross_val_score(LGBM_Regression, X_Train, Y, scoring='neg_mean_squared_error',
                                                 cv=folds)
print("Light GBM Regression K-Fold MSE Scores:", LGBM_Regression_cv_NMSE_scores)

print("Light GBM Regression Model CV Score Average: ", np.average(LGBM_Regression_cv_scores))
k_fold_cv_avg.append(np.average(LGBM_Regression_cv_scores))

print("Light GBM Regression Model N-MSE Score Average: ", np.average(LGBM_Regression_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(LGBM_Regression_cv_NMSE_scores))

print("Light GBM Regression Model Predictions :\n", LGBM_Regression_Predictions)
print("\nLight GBM Regression Model Score:", LGBM_Regression_Score)

scores.append(LGBM_Regression_Score)
names.append('LightGBM')
# ---------------------------------------- LightGBM Regression Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- CatBoost Regression Model ---------------------------------------- #
CAt_Boost_Regression = CatBoostRegressor(verbose=0, n_estimators=80, random_state=0)
CAt_Boost_Regression.fit(X_Train, Y)

CAt_Boost_Regression_Score = CAt_Boost_Regression.score(X_Train, Y)
CAt_Boost_Regression_Predictions = CAt_Boost_Regression.predict(X_Test)

CAt_Boost_Regression_cv_scores = cross_val_score(CAt_Boost_Regression, X_Train, Y, scoring='r2', cv=folds)
print("Cat Boost Model K-Fold CV Scores:", CAt_Boost_Regression_cv_scores)

CAt_Boost_Regression_cv_NMSE_scores = cross_val_score(CAt_Boost_Regression, X_Train, Y,
                                                      scoring='neg_mean_squared_error', cv=folds)
print("Cat Boost Model K-Fold MSE Scores:", CAt_Boost_Regression_cv_NMSE_scores)

print("Cat Boost Model Model CV Score Average: ", np.average(CAt_Boost_Regression_cv_scores))
k_fold_cv_avg.append(np.average(CAt_Boost_Regression_cv_scores))

print("Cat Boost Model N-MSE Score Average: ", np.average(CAt_Boost_Regression_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(CAt_Boost_Regression_cv_NMSE_scores))

print("Cat Boost Regression Model Predictions :\n", CAt_Boost_Regression_Predictions)
print("\nCat Boost Regression Model Score:", CAt_Boost_Regression_Score)

scores.append(CAt_Boost_Regression_Score)
names.append('CatBoost')
# ---------------------------------------- CatBoost Regression Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Ridge Model ---------------------------------------- #
Ridge = Ridge(random_state=0)
Ridge.fit(X_Train, Y)

Ridge_Score = Ridge.score(X_Train, Y)
Ridge_Predictions = Ridge.predict(X_Test)

Ridge_cv_scores = cross_val_score(Ridge, X_Train, Y, scoring='r2', cv=folds)
print("Ridge Model K-Fold CV Scores:", Ridge_cv_scores)

Ridge_cv_NMSE_scores = cross_val_score(Ridge, X_Train, Y, scoring='neg_mean_squared_error', cv=folds)
print("Ridge Model K-Fold MSE Scores:", Ridge_cv_NMSE_scores)

print("Ridge Model Model CV Score Average: ", np.average(Ridge_cv_scores))
k_fold_cv_avg.append(np.average(Ridge_cv_scores))

print("Ridge Model N-MSE Score Average: ", np.average(Ridge_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Ridge_cv_NMSE_scores))

print("Ridge Model Predictions :\n", Ridge_Predictions)
print("\nRidge Model Score:", Ridge_Score)

scores.append(Ridge_Score)
names.append('Ridge')
# ---------------------------------------- Ridge Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Lasso Model ---------------------------------------- #
Lasso_Model = Lasso(tol=0.10, random_state=0)
Lasso_Model.fit(X_Train, Y)

Lasso_Model_Score = Lasso_Model.score(X_Train, Y)
Lasso_Model_Predictions = Lasso_Model.predict(X_Test)

Lasso_Model_cv_scores = cross_val_score(Lasso_Model, X_Train, Y, scoring='r2', cv=folds)
print("Lasso Model K-Fold CV Scores:", Lasso_Model_cv_scores)

Lasso_Model_cv_NMSE_scores = cross_val_score(Lasso_Model, X_Train, Y, scoring='neg_mean_squared_error', cv=folds)
print("Lasso Model K-Fold MSE Scores:", Lasso_Model_cv_NMSE_scores)

print("Lasso Model Model CV Score Average: ", np.average(Lasso_Model_cv_scores))
k_fold_cv_avg.append(np.average(Lasso_Model_cv_scores))

print("Lasso Model N-MSE Score Average: ", np.average(Lasso_Model_cv_NMSE_scores))
k_fold_n_mse_avg.append(np.average(Lasso_Model_cv_NMSE_scores))

print("Lasso Model Predictions :\n", Lasso_Model_Predictions)
print("\nLasso Model Score:", Lasso_Model_Score)

scores.append(Lasso_Model_Score)
names.append('Lasso')
# ---------------------------------------- Lasso Model ---------------------------------------- #

print("----------------------------------------\n")

# ---------------------------------------- Plotting Accuracy Values ---------------------------------------- #
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gold', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:purple']
color_counter = 0

plt.figure("Non-Validated Scores")
plt.title('Accuracy Ratings')
plt.xlabel('Accuracy Scores')

while color_counter < len(colors):
    plt.barh(names[color_counter], scores[color_counter], color=colors[color_counter])
    color_counter += 1

# ---------------------------------------- Plotting Accuracy Values ---------------------------------------- #

print("----------------------------------------\n")

# ---------- Comparison between k-fold Cross Validation values to accuracy values ---------- #

# reset variable #
color_counter = 0
# reset variable #

plt.figure("Validated Scores")
plt.title('Cross Validation Scores')

while color_counter < len(k_fold_cv_avg):
    plt.barh(names[color_counter], k_fold_cv_avg[color_counter], color=colors[color_counter])
    color_counter += 1

plt.show()
# ---------- Comparison between k-fold Cross Validation values to accuracy values ---------- #

# Selected Prediction Values to Excel #
"""
XGB_Regression_Predictions = XGB_Regression_Predictions.reshape(-1, 1)
print(XGB_Regression_Predictions)
print(XGB_Regression_Predictions.shape)

DF = pd.DataFrame(XGB_Regression_Predictions)
DF.to_excel('test.xlsx', index=0, header=0)
"""
# Selected Prediction Values to Excel #

# ---------------------------------------------- END OF PROJECT ----------------------------------------------#
