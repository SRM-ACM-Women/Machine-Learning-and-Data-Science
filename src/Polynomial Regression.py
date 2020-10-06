import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('Position_Salaries.csv')
data.drop('Position', axis = 1, inplace = True)
X = data.iloc[:, :-1]
Y = data.iloc[:, 1]
#plotting the data
sns.scatterplot(x = X['Level'], y = Y, color = 'red')



#fitting the data on linear regression
from sklearn.linear_model import LinearRegression

first_regressor = LinearRegression()
first_regressor.fit(X, Y)

predictions = first_regressor.predict(X)



#fitting polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

second_regressor = LinearRegression()
second_regressor.fit(X_poly, Y)

plt.plot(X,predictions)


predictions2 = second_regressor.predict(X_poly)
X_grid = np.arange(1, 10, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,color = 'purple')
plt.plot(X_grid, second_regressor.predict(poly_reg.fit_transform(X_grid)), color = 'red')



#predicting a new value in linear regression
x = [6.5]
x = np.array(x)
x = x.reshape(len(x), 1)
first_regressor.predict(x)

#predicting a new value in polynomial regression
x_d = [6.5]
x_d = np.array(x_d)
x_d = x_d.reshape(len(x), 1)
second_regressor.predict(poly_reg.fit_transform(x_d))


