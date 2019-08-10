# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:17:07 2019

@author: Alex
"""
#import das lib necessarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#leitura da base
eleicao = pd.read_csv('Eleicao.csv', sep = ';')
#base de previsao
previsao = pd.read_csv('NovosCandidatos.csv', sep = ';')
#grafico 
plt.scatter(eleicao.DESPESAS, eleicao.SITUACAO)
eleicao.describe()
correlacao = np.corrcoef(eleicao.DESPESAS, eleicao.SITUACAO)
#preparando dados para regressao
X = eleicao.iloc[:, 2].values
X = X[:, np.newaxis]
y = eleicao.iloc[:, 1].values
#criando modelo de regressao logistica
modelo = LogisticRegression()
modelo.fit(X, y)
modelo.coef_
modelo.intercept_
#visualizando grafico de aprendizagem
plt.scatter(X, y)
Z = np.linspace(10, 3000, 100)
def model(x):
    return 1 / (1 + np.exp(-x))
r = model(Z * modelo.coef_ + modelo.intercept_).ravel()
plt.scatter(Z, r, color = 'red')
#novas previsoes
despesas = previsao.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)
executando_previsao = modelo.predict(despesas)


