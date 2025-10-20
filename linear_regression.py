# This code performs a basic linear regression on the car value data contained
# in the "car_value_vs_age.csv" file. Each of the data rows has only one response
# variable (car value in thousands of USD) and one predictor variable (car age
# in years). The standard gradient descent algorithm is used.

# Author: Your name here
# Date created: 
# Date modified:


# Set matplotlib backend to make sure it displays from the terminal
import matplotlib
# matplotlib.use('TkAgg') # This line won't run in github so comment it out before committing

# Read in necessary packages
import pandas as pd
import numpy as np
import seaborn as sns # plotting package
import matplotlib.pyplot as plt # plotting package

# Read in car value data from csv file
df = pd.read_csv("car_value_vs_age.csv")

# Graph scatterplot of data
sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age_years', y='Value_thousands_USD', data=df, s=100, color='purple', edgecolor='black')
plt.title('Car Value vs Age')
plt.xlabel('Age (years)')
plt.ylabel('Value (thousands USD)')

# plt.show() # comment this in ONLY when you want to see the plot of the data, rest of the code won't run while plot is displaying

# Fit data

x = df['Age_years'].to_numpy(dtype=np.float64) # array of x values
y = df['Value_thousands_USD'].to_numpy(dtype=np.float64) # array of y values

# initial guesses for alpha and beta
alpha = 0.
beta = 0.

## YOUR CODE HERE


## DEFINE GRADIENT CALCULATING FUNCTION

def grad(x, y, alpha, beta):
    """

    Calculate gradient of sum of square errors (SSE) loss function

    Inputs: arrays of x and y values, scalar parameters alpha and beta
    Output: array containing partial derivatives of loss function w.r.t.
            linear regression parameters alpha and beta

    """
    ## YOUR CODE HERE
    partial_alpha = np.sum(x * (alpha * x + beta - y))
    partial_beta = np.sum(alpha * x + beta - y)

    return np.array([partial_alpha, partial_beta])


# perform gradient descent with (optional) iteration check


## MORE OF YOUR CODE HERE
learning_rate = 0.0001
tolerance = 1e-8
max_iterations = 10000

params = np.array([alpha, beta])

iteration = 0
while True:

    gradient = grad(x, y, params[0], params[1])

    # calculate the magnitude of the gradient
    grad_norm = np.linalg.norm(gradient)

    if grad_norm < tolerance:
        break

    
    if iteration >= max_iterations:
        break

    params -= learning_rate * gradient

    iteration += 1



print("Converged parameters:", params)


# Overlay regression line onto scatterplot
x_vals = np.linspace(df['Age_years'].min(), df['Age_years'].max(), 100)  # fine grid for smooth line
y_vals = params[0] * x_vals + params[1]
plt.plot(x_vals, y_vals, color='red', linewidth=2, label=f'y = {params[0]:.2f}x + {params[1]:.2f}')

# Labels and title
plt.title('Car Value vs Age with Regression Line')
plt.xlabel('Age (years)')
plt.ylabel('Value (thousands USD)')
plt.legend()
plt.savefig('car_value_vs_age_regression_1.png')
# plt.show() COMMENT OUT if running pytest

