# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:03:07 2020

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

%matplotlib inline

df = pd.read_csv('dados/usa_housing.csv')
print(df.head())
print(df.info())
print(df.columns)

# análises iniciais
sns.pairplot(df)
sns.heatmap(df.corr())

# campos preditores
# nao utilizamos adress pois o mesmo e uma string
x = df[['Avg. Area Income',
       'Avg. Area House Age',
       'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms',
       'Area Population']]

# classe
y = df['Price']

# separando a base de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state = 99)

# criando e treinando o modelo
model = LinearRegression()
model.fit(x_train, y_train)

# intercept
print(model.intercept_)
# coeficientes
print(model.coef_)
coef = pd.DataFrame(model.coef_, x.columns, columns=['Coefs'])

# predições de teste
pred = model.predict(x_test)

# grafico comparativo das predições vs real
plt.scatter(y_test, pred)

# grafico de distribuição dos erros
sns.distplot((y_test - pred))

# metricas de avaliação
# MAE - erro absoluto médio
mae = metrics.mean_absolute_error(y_test, pred)
print(f'MAE: {mae}')
# MSE - média dos erros quadrados
mse = metrics.mean_squared_error(y_test, pred)
print(f'MSE: {mse}')
# RMSE - raiz quadrada da media dos erros quadrados
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')