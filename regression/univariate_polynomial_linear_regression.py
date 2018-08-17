# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Position_Salaries.csv')
print(dataset.head())
X = dataset.iloc[:, 1:-1].values# We only need the level column
y = dataset.iloc[:, -1].values
X_original = dataset.iloc[:, 1:-1].values

# Add quadratic term to X
X = np.hstack([X, X ** 2])

'''
    The above step which was to append the polynomial data to X could
    be done by sklearn too.

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree = 2) # Specify the required degree here
    X_poly = poly.fit_transform(X)

    # This method also adds the bias by itself.
'''
# Account for the bias term
X = np.hstack([np.ones((X.shape[0], 1)), X])

'''
    Since we have only a few number of data, we will use the whole set to train the model
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
pred_sk = regressor.predict(X)

w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
pred = X.dot(w)

plt.scatter(X[:, 1], y, color = 'red')
plt.plot(X[:, 1], pred_sk, color = 'green')
plt.plot(X[:, 1], pred, color = 'blue')
plt.title('Quadratic model')
plt.show()

regressor_linear = LinearRegression()
regressor_linear.fit(X_original, y)
pred_lr = regressor_linear.predict(X_original)

plt.title('Linear model')
plt.scatter(X_original, y, color = 'red')
plt.plot(X_original, pred_lr, color = 'blue')
plt.show()

