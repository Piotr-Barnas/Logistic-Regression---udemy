# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:46:25 2019

@author: ASUS
"""

import pandas as pd

df = pd.read_csv('advertising.csv')

print(df.head())
print(df.info())
print(df.describe())

from sklearn.model_selection import train_test_split
print(df.columns)
X = df[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(X_train, y_train)

predictions = logReg.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))