# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1, 1))

# Fit SVR to the dataset
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X, Y)

#Visualize SVR
plt.scatter(X, Y)
plt.plot(X, svr.predict(X), color = 'red')
plt.title('Truth or Bluff (SVR)')
plt.ylabel('Salary')
plt.xlabel('Position level')
plt.show()

#Predict the salary for position level of 6.5
Y_prediction = int(sc_Y.inverse_transform(svr.predict(sc_X.transform(np.array([[6.5]])))))
print("SVR Prediction for Position Level of 6.5: $" + str(Y_prediction))


# Visualising the SVR results (for higher resolution and smoother curve)
"""
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, svr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""