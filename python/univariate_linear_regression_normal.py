# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
dataset = dataset.as_matrix()
X = dataset[:, :-1]
y = dataset[:, -1]

# Adding an array of ones to X to account for the bias term
X = np.hstack([np.ones((X.shape[0], 1)), X])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Predicting using sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
predictions_sklearn = regressor.predict(x_test)

# Predicting using normal equation
w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
test_predictions = x_test.dot(w)
# print(test_predictions, y_test)

plt.scatter(x_test[:, 1], y_test, color = 'blue')
plt.plot(x_test[:, 1], predictions_sklearn, color = 'red')
plt.plot(x_test[:, 1], test_predictions, color = 'green')

plt.show()
error = np.mean(np.abs(test_predictions - predictions_sklearn))*100
print("Error between predictions is: {} %".format(error) )
