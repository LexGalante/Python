# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:19:54 2019

@author: Alex
"""
#import das lib necessarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
import statsmodels.formula.api as sm
#leitura da base
base = pd.read_csv('mt_cars.csv')
#remoção de atributo desnecessario adicionado pelo pandas
base = base.drop(['Unnamed: 0'], axis = 1)
#verificando correlação entre consumo e polegadas
X = base.iloc[:, 2].values
y =base.iloc[:, 0].values
correlacao = np.corrcoef(X, y)
#preparando reshape par criar o modelo
X = X.reshape(-1, 1)
#criando o modelo
modelo = LinearRegression()
modelo.fit(X, y)
#verificando informacoes do modelo
modelo.intercept_
modelo.coef_
#verificando o coeficiente de determinação, correlação ao quadrado
modelo.score(X, y)
#verificando o coeficiente de determinação ajustado, correlação ao quadrado
previsoes = modelo.predict(X)
modelo_ajustado = sm.ols(formula = 'mpg ~ disp', data = base)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()
#visualizando o grafico
plt.scatter(X, y)
#adicionando linha de regressao
plt.plot(X, previsoes, color = 'red')
#verificando previsao
modelo.predict(200)
#preparando regressao multila
X1 = base.iloc[:, 1:4].values
y1 = base.iloc[:, 0].values
modelo1 = LinearRegression()
modelo1.fit(X1, y1)
modelo.score(X1, y1)
modelo_ajustado1 = sm.ols(formula = 'mpg ~ disp + hp + cyl', data = base)
modelo_treinado1 = modelo_ajustado1.fit()
modelo_treinado1.summary()
#previsao
novo = np.array([4, 200, 100])
novo = novo.reshape(1, -1)
modelo1.predict(novo)