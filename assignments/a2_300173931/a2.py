# CSI4106 - Assignment 2: Machine Learning
'''Name**: Raj Badial 
**Student Number**: 300173931

Shereen Etemad
**Student Number**: 300186291

**Group Members**: Raj Badial, Shereen Etemad, Group 30 
**Task Division**: 
-
-
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV



# 1 loading the dataset
url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a2/diabetes_prediction_dataset.csv'
df = pd.read_csv(url)
print(df.head())

# 2 Seeing unique values and seeing distribution using histograms and boxplots

# Display unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("\n")

print(df.describe(include='all'))

# Create histograms for all features
df.hist(figsize=(10, 8), bins=20)
plt.show()

# Create boxplots to detect outliers for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# 3 Target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='diabetes', data=df)
plt.title('Target Variable Distribution')
plt.show()

# 4 Splitting data into training and test using the holdout method (random_state)
X = df.drop(columns=['diabetes'])
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5 Encoding categorical columns (gender and smoking_history)

# using one-hot encoding for gender as the order of the items (male, female and other) don't have any effect and should be treated as separate entities
# do it for both training and test data
X_train = pd.get_dummies(X_train, columns=['gender'])
X_test = pd.get_dummies(X_test, columns=['gender'])

# Manual label encoding for 'smoking_history' so I have control over the mapping of the possible values and can ensure its in the proper order
# Using label encoding as order here does matter and should be considered
# not current and former could be switched, we were unsure on which should have precedence in matters of order
smoking_mapping = {
    'No Info': 0,
    'never': 1,
    'ever': 2,
    'former': 3,
    'not current': 4,
    'current': 5
}

# Apply the mapping to training and test
X_train['smoking_history_encoded'] = X_train['smoking_history'].map(smoking_mapping)
X_test['smoking_history_encoded'] = X_test['smoking_history'].map(smoking_mapping)

# Drop the original columns
X_train = X_train.drop(columns=['smoking_history'])
X_test = X_test.drop(columns=['smoking_history'])


# 6 Standardization of data

scaler = StandardScaler()
X_train[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.fit_transform(X_train[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
X_test[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(X_test[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])

# 7 Model Development

# Step 1: Initializing the models with default parameters
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Step 2: Fit the models on the training data
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Step 3: Make predictions on the test data
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# 8 Model Eval - using F1 score, precision and recall to evaluate each model

# Define scoring metrics to calculate during cross-validation
scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# Perform cross-validation for Decision Tree
cv_dt = cross_validate(dt_model, X_train, y_train, cv=5, scoring=scoring)
print(f"Decision Tree CV Average Precision: {cv_dt['test_precision'].mean():.4f}")
print(f"Decision Tree CV Average Recall: {cv_dt['test_recall'].mean():.4f}")
print(f"Decision Tree CV Average F1-Score: {cv_dt['test_f1'].mean():.4f}")

# Perform cross-validation for KNN
cv_knn = cross_validate(knn_model, X_train, y_train, cv=5, scoring=scoring)
print(f"\nKNN CV Average Precision: {cv_knn['test_precision'].mean():.4f}")
print(f"KNN CV Average Recall: {cv_knn['test_recall'].mean():.4f}")
print(f"KNN CV Average F1-Score: {cv_knn['test_f1'].mean():.4f}")

# Perform cross-validation for Logistic Regression
cv_lr = cross_validate(lr_model, X_train, y_train, cv=5, scoring=scoring)
print(f"\nLogistic Regression CV Average Precision: {cv_lr['test_precision'].mean():.4f}")
print(f"Logistic Regression CV Average Recall: {cv_lr['test_recall'].mean():.4f}")
print(f"Logistic Regression CV Average F1-Score: {cv_lr['test_f1'].mean():.4f}")

# Step 9. Hyperparameter Optimization

# Defining the hyperparameter grids for each model
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, None]
}

knn_param_grid = {
    'n_neighbors': [3, 5],
    'weights': ['uniform', 'distance']
}

lr_param_grid = {
    'penalty': ['l2'],
    'max_iter': [1000],
    'tol': [1e-4]
}

# Setting up GridSearchCV for each model

# Grid search for Decision Tree
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, scoring='f1', refit=True)
dt_grid_search.fit(X_train, y_train)

# Grid search for KNN
knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='f1', refit=True)
knn_grid_search.fit(X_train, y_train)

# Grid search for Logistic Regression
lr_grid_search = GridSearchCV(LogisticRegression(random_state=42), lr_param_grid, cv=5, scoring='f1', refit=True)
lr_grid_search.fit(X_train, y_train)

# Extracting the best hyperparameters and F1-scores
best_params_dt = dt_grid_search.best_params_
best_params_knn = knn_grid_search.best_params_
best_params_lr = lr_grid_search.best_params_

best_f1_dt = dt_grid_search.best_score_
best_f1_knn = knn_grid_search.best_score_
best_f1_lr = lr_grid_search.best_score_

# Creating a summary dataframe
results = pd.DataFrame({
    'Model': ['Decision Tree', 'KNN', 'Logistic Regression'],
    'Best Params': [best_params_dt, best_params_knn, best_params_lr],
    'Best F1-Score': [best_f1_dt, best_f1_knn, best_f1_lr]
})

# Display the best results
print("Best Hyperparameters and F1-Scores:")
print(results)

# Visualizing the performance of hyperparameter combinations
# Extracting mean test scores from GridSearchCV results
dt_scores = dt_grid_search.cv_results_['mean_test_score']
knn_scores = knn_grid_search.cv_results_['mean_test_score']
lr_scores = lr_grid_search.cv_results_['mean_test_score']

# Plotting F1-scores for different hyperparameter combinations
plt.figure(figsize=(15, 5))

# Decision Tree plot
plt.subplot(1, 3, 1)
plt.bar(range(len(dt_scores)), dt_scores)
plt.title('Decision Tree F1-Score by Hyperparameter')
plt.xlabel('Parameter Combination')
plt.ylabel('F1-Score')

# KNN plot
plt.subplot(1, 3, 2)
plt.bar(range(len(knn_scores)), knn_scores)
plt.title('KNN F1-Score by Hyperparameter')
plt.xlabel('Parameter Combination')

# Logistic Regression plot
plt.subplot(1, 3, 3)
plt.bar(range(len(lr_scores)), lr_scores)
plt.title('Logistic Regression F1-Score by Hyperparameter')
plt.xlabel('Parameter Combination')

plt.tight_layout()
plt.show()

# Summary of cross-validation performance
cv_results = {
    'Model': ['Decision Tree', 'KNN', 'Logistic Regression'],
    'CV F1-Score': [0.8015, 0.7255, 0.7284]
}
cv_df = pd.DataFrame(cv_results)

# Summary of test performance
test_results = {
    'Model': ['Decision Tree', 'KNN', 'Logistic Regression'],
    'Precision': [0.986, 0.896, 0.867],
    'Recall': [0.682, 0.612, 0.613],
    'F1-Score': [0.806, 0.728, 0.718]
}
test_df = pd.DataFrame(test_results)

# Display the results
print("Cross-Validation Results:")
print(cv_df)
print("\nTest Data Results:")
print(test_df)

# Analysis of the differences between models
"""
- The Decision Tree performed best overall, particularly in terms of precision (0.986), but it has a lower recall (0.682), 
  meaning it missed some positive cases.
- KNN had a more balanced performance but slightly lower F1-score (0.728).
- Logistic Regression was close to KNN with an F1-score of 0.718, offering a simpler and robust solution with solid precision and recall.

Recommendation:
- If precision is the priority, go with the Decision Tree.
- If recall or a more balanced model is needed, KNN or Logistic Regression are strong candidates.
- Logistic Regression is recommended if generalization to unseen data is critical, due to its simplicity and robustness.
"""


# Resources
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://www.atlassian.com/data/charts/box-plot-complete-guide#:~:text=Box%20plots%20are%20used%20to,skew%2C%20variance%2C%20and%20outliers.
# https://scikit-learn.org/1.5/modules/preprocessing.html
# https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://www.w3schools.com/python/python_ml_cross_validation.asp
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html



