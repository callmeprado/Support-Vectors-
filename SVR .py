#Support Vector Regression 

#importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing Data set
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values #selecting one col for x, but we dont need a vector of x, We need a Matrix.
Y = dataset.iloc[:,2].values

plt.scatter(dataset.Level, dataset.Salary)
plt.show()

#NO Splitting into Train & Test due to insufficient AMOUNT of data (only 10 observations in this case)
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0 )"""

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = Y.reshape(-1,1)
Y = sc_y.fit_transform(Y)



#FITTING SVR Model to Dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predicting new result with Regression 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Vizualizing Polynomial Regression results 

plt.scatter(X, Y, color = 'Red')
plt.plot(X, regressor.predict(X), color = 'Blue')
plt.title('Support Vector Regression results')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()