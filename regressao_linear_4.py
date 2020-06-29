# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:31:04 2020

@author: alexg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from yellowbrick.regressor import ResidualsPlot

# leitura do dataset completo
df = pd.read_excel('dados/covid19_br.xlsx')

# separando os dados do gerais do Brasil
brasil = df[df['regiao'] == 'Brasil']
# os primeiros 9 dados do dataset
print(brasil.head())
# informações uteis sobre o dataset
print(brasil.info())

"""
TRATAMENTOS
Aqui vamos preparar nossa base, como queremos analisar apenas dados do BRASIL
vamos filtra-los
"""
brasil.drop(['regiao',
             'estado',
             'municipio',
             'coduf',
             'codmun',
             'codRegiaoSaude',
             'nomeRegiaoSaude',
             'FgMetro'], axis=1, inplace=True)
# preenchendo os valores NAN com 0
brasil = brasil.fillna(value=0)
# checando se o campo DATA está como um tipo timestamp do python
print(type(brasil['data'][0]))

"""
ANÁLISES EXPLORATÓRIAS
Aqui faremos uma serie de análise sobre no novo dataset
Correlações entre as variaveis
Ultimo dia analisado
Total de casos
Total de mortos
"""
# correlações entre variaveis da dataframe
sns.heatmap(brasil.corr())
# uma série de gráficos exploratorios entre as variaveis do dataframe
sns.pairplot(brasil)
# após essa análise é nitida a forte correlação entre:
# casosAcumulado e obitosAcumulado
# vamos preparar um gráfico mais especifico sobre estas duas variaveis
sns.lmplot(x='casosAcumulado', y='obitosAcumulado', data=brasil)

# ultimo dia da analise realizada
last_day = brasil['data'].max().strftime('%d/%m/%Y')
cases = brasil['casosAcumulado'].max()
deaths = brasil['obitosAcumulado'].max()

print(f'Total de casos confirmados em {last_day}: {cases}')
print(f'Total de mortes confirmadas em {last_day}: {deaths}')

"""
PREPARANDO DADOS DE TREINO E TESTE
Nessa etapa o intuito e preparar nosso X e Y, bem como nossos dados de treino 
e teste
"""
X = brasil['casosAcumulado'].values
X = X.reshape(-1, 1)
Y = brasil['obitosAcumulado'].values

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2, 
                                                    random_state = 99)

"""
CRIANDO E TREINANDO O MODELO
Aqui no vamos criar o modelo que é um objeto LinearRegression do sklearn
Recomendo a leitura da documentação para entendimento das variaveis e suas 
possibilidades
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""
model = LinearRegression()
model.fit(x_train, y_train)
# detalhes do modelo: interceptor
print(model.intercept_)
# detalhes do modelo: coeficientes
print(model.coef_)

"""
PREDIÇÕES
"""
# prevendo a base de teste
predicts = model.predict(x_test)
# prevendo dados inputados
# informe o total de casos no dia e printa a previsão do total de mortes
new_amount_cases = float(input('Total de casos no dia: '))
print(model.predict(new_amount_cases))


"""
ANÁLISES DOS ERROS
"""
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
# distribuição dos erros
sns.distplot((y_test - predicts))
# gráfico yellowbrick
residuals = ResidualsPlot(model)
residuals.fit(X, Y)
residuals.poof()