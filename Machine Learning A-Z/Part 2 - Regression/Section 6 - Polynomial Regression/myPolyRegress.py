# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:04:32 2017

@author: schnebly
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X, Y)

#View Linear Regression
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg1.predict(X), color = "blue")
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# View the Polynomial Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

# Linear prediction of salary for position level of 6.5 
linearPrediction = int(lin_reg1.predict(6.5))
print("Linear Prediction for Position level 6.5 : $" + str(linearPrediction))

#Polynomial prediction of salary for position level of 6.5
polyPrediction = int(lin_reg2.predict(poly_reg.fit_transform(6.5)))
print("Polynomial Prediction for Position level 6.5 : $" + str(polyPrediction))

















