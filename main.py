import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the heart data
df = pd.read_csv('heartdata.csv')

# print the first 5 rows of the data set
print(df.head())

# determine the datatype of each column
print(df.dtypes)

# print the unique values for the columns **ca** and **thal**
print(df.ca.unique())
print(df.thal.unique())

# determine how many rows contain missing values in the columns **ca** and **thal**
print(df.ca.isnull().sum())
print(df.thal.isnull().sum())

# determine how many rows contain missing values, the python code is below
len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

# since only 6 rows have missing values, let's look at them
print(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

# count the number of rows in the full dataset
print(len(df))

# remove the rows with missing values
df = df.dropna()

# verify that the rows with missing values have been removed
print(len(df))

# verify using the unique function that "ca" and "thal" do not have missing values
print(df.ca.unique())
print(df.thal.unique())

# split the data into dependent and independent variables
# the column of data that we will to to make classifications
X = df.iloc[:,:-1] # this line of code is the same as X = df.drop(['ca'], axis=1)
# the column of data that we want to predict
y = df.iloc[:,-1] # this line of code is the same as y = df['ca']

# print the head of both the X and y dataframes so that you can verify this worked correctly
print(X.head())
print(y.head())

