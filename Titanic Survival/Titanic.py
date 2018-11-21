# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:19:30 2018

@author: sunka
"""

import pandas as pd

data = pd.read_csv('train.csv')
print(data)
test = pd.read_csv('test.csv')
print(test)

data.columns
test.columns

data.isnull().any()

data = data.fillna(0)

data.isnull().any()

from sklearn import preprocessing

encode = preprocessing.LabelEncoder()

enc = encode.fit_transform(data['Sex'])
ence = encode.fit_transform(data['Name'])

y = data[ 'Survived']

df1 = data['Sex']
print(df1)

j = 0
for i in df1:
    if i == 'male':
        df1[j] = 1
    else:
        df1[j] = 0
    j+=1
    
print(df1)

data['Sex'] = df1

data.columns

data = data[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare']]

x = data[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare']]

test.columns

test = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

df1 = test['Sex']
print(df1)
j = 0
for i in df1:
    if i == 'male':
        df1[j] = 1
    else:
        df1[j] = 0
    j+=1
    
print(df1)

test['Sex'] = df1

test.isnull().any()

test = test.fillna(0)

from sklearn import tree

trees = tree.DecisionTreeClassifier()
fitting = trees.fit(x,y)
result = trees.predict(test)

from matplotlib import pyplot

pyplot.hist(result, color = 'orange')