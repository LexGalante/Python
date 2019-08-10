#import das lib necessarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
#leitura da base
base = pd.read_csv('cars.csv')
#remoção de atributo desnecessario adicionado pelo pandas
base = base.drop(['Unnamed: 0'], axis = 1)
#atribuindo x e y
X = base.iloc[:, 1].values
y = base.iloc[:, 0].values
#calculando a correlação
correlacao = np.corrcoef(X, y)
#criando modelo de regressao linear
X = X.reshape(-1, 1)
modelo = LinearRegression()
modelo.fit(X, y)
#verificando informacoes do modelo
modelo.intercept_
modelo.coef_
#criando o grafico de dispersao
plt.scatter(X, y)
#adicionando a linha de regressao
plt.plot(X, modelo.predict(X), color = 'red')
#prevendo valores de parada de 22
previsao = modelo.intercept_ + modelo.coef_ * 22
#prevendo usando funcao
previsao = modelo.predict(22)
#visualizando residuais do modelo
modelo._residues
#utilizando yellow brick para visualizacao
visualizador = ResidualsPlot(modelo)
visualizador.fit(X, y)
visualizador.poof()