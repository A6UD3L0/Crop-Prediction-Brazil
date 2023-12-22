#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################################
#							Script: SuperLearner.py
#							Author: Agudelo
#							Date: May 5, 2022
#######################################################################################

"""
Description:
This script demonstrates the use of a SuperLearner ensemble with various base classifiers
to predict 'destinated_area' in a dataset. The base models include Logistic Regression,
Decision Tree, SVM, Naive Bayes, K-Nearest Neighbors, AdaBoost, Bagging, Random Forest,
and Extra Trees classifiers. The ensemble is evaluated on a validation set, and the
accuracy score is printed.

Note: Ensure that the 'Categorical_values' module and other required libraries are
correctly imported and available in your environment for the script to run successfully.
"""

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner
import Categorical_values as DF
import pandas as pd

# Load the data using the function from the module 'Categorical_values'
df = DF.asginar_variables_cat()

# Function to get a list of base models
def get_models():
    models = list()
    models.append(LogisticRegression(solver='liblinear'))
    models.append(DecisionTreeClassifier())
    models.append(SVC(gamma='scale', probability=True))
    models.append(GaussianNB())
    models.append(KNeighborsClassifier())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier(n_estimators=10))
    models.append(RandomForestClassifier(n_estimators=10))
    models.append(ExtraTreesClassifier(n_estimators=10))
    return models

# Function to create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X))
    # Add base models to the ensemble
    models = get_models()
    ensemble.add(models)
    # Add the meta-model (Logistic Regression in this case)
    ensemble.add_meta(LogisticRegression(solver='lbfgs'))
    return ensemble

# Prepare inputs and outputs
df = df.dropna()
X = df.drop(['LN_destinated_area', 'year', "city_code", "product_type", "product", "destinated_area"], axis=1)
X = X.dropna()
X = X.to_numpy()
y = df["destinated_area"]
y = y.dropna()
y = y.to_numpy()

# Split the data into training and validation sets
X, X_val, y, y_val = train_test_split(X, y, test_size=0.10)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)

# Create the super learner
ensemble = get_super_learner(X)

# Fit the super learner
ensemble.fit(X, y)

# Summarize base learners
print(ensemble.data)

# Make predictions on the validation set
yhat = ensemble.predict(X_val)
print('Super Learner Accuracy: %.3f' % (accuracy_score(y_val, yhat) * 100))
