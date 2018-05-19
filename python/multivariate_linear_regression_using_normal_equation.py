# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('datasets/50_Startups.csv')
print(dataset.head())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# We have a categorical data here. So, we need to preprocess the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lencoder = LabelEncoder() # Label Encoder is used to convert categorical data to unique integer values
X[:, -1] = lencoder.fit_transform(X[:, -1]) # Converts the locations to integral labels
oencoder = OneHotEncoder(categorical_features = [3])
X = oencoder.fit_transform(X).toarray()

# Remove a dummy variable column
X = X[:, 1:]
# Add all ones to the first column of X to account for the bias
X = np.hstack([np.ones((X.shape[0], 1)), X])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Predicting using sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
predictions_sklearn = regressor.predict(x_test)

# Predicting using normal equation
w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
test_predictions = x_test.dot(w)

error = np.mean(np.abs(test_predictions - predictions_sklearn)) * 100
print("Error between predictions of sklearn and normal equation is: {} %".format(error) )
