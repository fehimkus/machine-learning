# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:57:51 2020

@author: TUBA_BOZBAYIR
"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

pima.head()


#future selection split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
feature_cols2 = ['label']

y_old = pima.label # Target variable
y = pima[feature_cols2]


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


model = PolynomialFeatures(degree=4)
x_poly = model.fit_transform(X)




lg = LinearRegression()
lg.fit(x_poly, y)
y_pred_linear = lg.predict(X)
#print(X_test, y_pred, y)
print("linear regression başarı karşılaştırması",mean_squared_error(y_pred_linear, y_test))

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#optimizin dec tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Optimize accuracy",metrics.accuracy_score(y_test, y_pred))




