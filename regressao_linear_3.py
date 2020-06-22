# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:49:48 2020

@author: Alex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""
Uma empresa quer saber se deve investir mais no app ou no site
No dataset abaixo temos os seguintes dados
    Avg. Session Length: Tempo médio de sessão
    Time on App: Tempo de uso no aplicativo
    Time on Website: Tempo de uso do site
    Length of Membership: Tempo que pessoa é do programada de fidelidade
"""

df = pd.read_csv('dados/ecommerce_customer.csv')

print(df.head())
print(df.info())
print(df.describe())

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)

sns.pairplot(df)

# após analisar esse dataset inicialmente podemos dizer
# que o Length of Membership possui uma relação linear com Yearly Amount Spent
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)

# separando classe
print(df.columns)
x = df[['Avg. Session Length',
        'Time on App',
       'Time on Website',
       'Length of Membership']]
y = df['Yearly Amount Spent']

# separando treino e teste
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=22)

model = LinearRegression()
model.fit(x_train, y_train)

# predições
print(model.coef_)
predicts = model.predict(x_test)

# comparando o previsto X real
plt.scatter(y_test, predicts)
plt.xlabel('Real')
plt.ylabel('Previsto')

# calculando as metricas de erros do modelo
# MAE - erro absoluto médio
mae = metrics.mean_absolute_error(y_test, predicts)
print(f'MAE: {mae}')
# MSE - média dos erros quadrados
mse = metrics.mean_squared_error(y_test, predicts)
print(f'MSE: {mse}')
# RMSE - raiz quadrada da media dos erros quadrados
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')
sns.distplot((y_test - predicts))


coef = pd.DataFrame(model.coef_, x.columns, columns=['Coeficientes'])
print(coef)