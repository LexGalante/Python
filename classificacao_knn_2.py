# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:09:38 2020

@author: Alex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('dados/anonymous_data.csv', index_col=0)

print(df.head())
print(df.info())
print(df.columns)

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1))

normalize = scaler.transform(df.drop('TARGET CLASS', axis = 1))

df_normalized = pd.DataFrame(normalize, columns=df.columns[:-1])
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df_normalized,
                                                    df['TARGET CLASS'],
                                                    test_size=0.3,
                                                    random_state=99)

model = KNeighborsClassifier(n_neighbors=20)
model.fit(x_train, y_train)
preditcs = model.predict(x_test)
print(classification_report(y_test, preditcs))
print(confusion_matrix(y_test, preditcs))

error_rate = []
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    preditcs = model.predict(x_test)
    error_rate.append(np.mean(preditcs != y_test))

plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='+')
plt.xlabel('K')
plt.ylabel('Error')
plt.title('Best Numbers of K')
