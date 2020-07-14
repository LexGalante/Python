# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:41:24 2020

@author: Alex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# dados sobre crianças que apresentam problemas de corcundes
# processos cirurgicos que criança foi submetida
df = pd.read_csv('dados/kyphosis.csv')
print(df.head())

sns.pairplot(df, hue='Kyphosis')

x = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

simple_tree = DecisionTreeClassifier()
simple_tree.fit(x_train, y_train)
predicts = simple_tree.predict(x_test)
print(classification_report(y_test, predicts)) 
print(confusion_matrix(y_test, predicts))

forest = RandomForestClassifier(n_estimators=200)
forest.fit(x_train, y_train)
predicts_forest = forest.predict(x_test)
print(classification_report(y_test, predicts_forest)) 
print(confusion_matrix(y_test, predicts_forest))
