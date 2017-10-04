Adding in the Machine Learning

import numpy as np # For the mathematical Array computation and dimentional function
from sklearn.linear_model import LinearRegression # Having the Linear Regression model
import pandas as pd # For Reading csv or files from the web has functions like read_csv and dataframes + series
import matplotlib.pyplot as plt # For the ploting of graphs and visualization (kind of data visualization)
from numpy.linalg import inv # Linear Algorithms from numpy to have inverter function
from numpy import dot, transpose # For having a dot function for the Dot product faster and the transpose of matrix
from sklearn.preprocessing import PolynomialFeature # For having the Polynomials degree implementation on the datasets
from sklearn.model_selection import train_test_split # For having the sklearning module to have 4 data set train and test for                                                         both response and exploratory data variable.
from sklearn.cross_validation import cross_val_score # For having the multiple crossvalidation scores for the data
from sklearn.linear_model import SGDRegressor # Stochastic Gradient descent , has a parameters known as loss "squared_loss"
from sklearn.datasets import load_boston # Default data set provided by the sklearn
from sklearn.preprocessing import StandardScaler # For the standardize features by removing the mean and scaling to unit variance
