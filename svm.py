# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:17:15 2020

@author: Alex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


data = load_breast_cancer()
print(data.keys())
print(data['DESCR'])
print(data['feature_names'])

df_x = pd.DataFrame(data['data'], columns=data['feature_names'])
df_y = pd.DataFrame(data['target'], columns=['Cancer'])

x_train, x_test, y_train, y_test = train_test_split(df_x, np.ravel(df_y),
                                                    test_size=0.3,
                                                    random_state = 99)

model = SVC()
model.fit(x_train, y_train)

predicts = model.predict(x_test)
print(classification_report(y_test, predicts))

print(confusion_matrix(y_test, predicts))
# metodo para testar vários parametros de entrada no SVM
grid_search = GridSearchCV(SVC(), 
                           {
                               'C': [0.1,1,10,100,1000],
                               'gamma': [1, 0.1, 0.001, 0.0001],
                               'kernel': ['rbf']
                           }, verbose=3)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)

predicts = grid_search.predict(x_test)
print(classification_report(y_test, predicts))
print(confusion_matrix(y_test, predicts))
