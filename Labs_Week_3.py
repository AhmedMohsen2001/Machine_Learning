import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

def perceptron_init():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
    y_train = np.array([0, 0, 0, 1, 1, 1])                                       #(m,)
    