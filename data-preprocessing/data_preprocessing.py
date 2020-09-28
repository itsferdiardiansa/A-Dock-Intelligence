# Data preprocesssing
# @author <ferdiardiansa@gmail.com>

# The libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # encoding categorical
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Dataset
dataset = pd.read_csv('data-preprocessing/Data.csv') 
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Missing Data
imputer = Imputer()
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Categorical Data
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = ColumnTransformer([('encode', OneHotEncoder(), [0])], remainder = 'passthrough')
x = onehotencoder.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)