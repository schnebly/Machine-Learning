# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Multiple_Linear_Regression/50_startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#encoding categorical data in X into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoid the dummy var trap
X = X[:,1:]

#split data into test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = .2)

#fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the Test set of the results
y_prediction = regressor.predict(X_test)


################################################
#building optimal model using Backward Elimination with sig level = .05S
import statsmodels.formula.api as sm

#add column of b_0(1's) to X
X = np.append(arr =  np.ones((50,1)).astype(int), values = X, axis = 1)

##repeat this block of code for every P < sig level starting from all dependant variables
X_optimal = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
X_optimal = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
X_optimal = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
X_optimal = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
X_optimal = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()

##copy paste this line to test
regressor_OLS.summary()