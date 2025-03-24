# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jayasree T S
RegisterNumber: 24900147
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('///content/student_scores ml.csv')
data.head()
data.tail()
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state = 42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print("Mean Square Error: ", mse)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)
rmse = np.sqrt(mse)
print("Root Mean Square Error: ",rmse)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

```
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![ml exp 2](https://github.com/user-attachments/assets/05fcee01-92cf-4037-8068-6bade7c0bd67)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
