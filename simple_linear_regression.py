# program for imlenmenting simple Linear Regression
# Start by importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import the dataset the code should be in thesame folder also in same runtime if you are in google colab
dataset = pd.read_csv('name_of_dataset.csv')
#now select every row and every column except the last column .Last column being our dependant column
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#Splitting the dataset as Training set and Test set .Training using the entire data may cause oevrfitting 
from sklearn.model_selection import train_test_split
#test_size =1/3 or 30% implies that we are using only 30% of data as test data ( 20% is also preferred )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#from scikit learn call linear regression model 
from sklearn.linear_model import LinearRegression
#creating object of the class 
regressor = LinearRegression()
#fit function fits the traininig data to the object
regressor.fit(X_train, y_train)
#Predicting the  results
y_pred = regressor.predict(X_test)
# Visualising the Training set results
#assumes that we are dealing with salary data 
plt.scatter(X_train, y_train, color = 'red')
#plots a line 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Visualising test set results 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()