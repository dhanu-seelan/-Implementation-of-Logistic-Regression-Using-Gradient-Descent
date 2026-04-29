# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select the required input features (ssc_p, mba_p) and target variable (status). Convert categorical output values into numeric form (Placed = 1, Not Placed = 0).
2. Normalize the input data using StandardScaler() and add a bias column of ones to the feature matrix for the intercept term.
3. Initialize weights (theta) with zeros and apply the sigmoid function. Use Gradient Descent to update weights repeatedly by minimizing the logistic regression cost function.
4. Predict the output using the trained weights, calculate the accuracy of the model, and plot the graph of Cost vs Iterations to observe convergence.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Danaseelan G
RegisterNumber: 212225040053
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data.csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)


accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()

```
## Output:
<img width="945" height="735" alt="image" src="https://github.com/user-attachments/assets/7a0818ed-bac5-4f5e-92db-05baa3c3f559" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

