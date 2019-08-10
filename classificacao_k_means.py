# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:09:01 2019

@author: Alex
"""
from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#definido o dataset
iris = datasets.load_iris()
#dimensoes do dataset
classe, quantidade = np.unique(iris.target, return_counts = True)
#criando o modelo
modelo = KMeans(n_clusters=3)
modelo.fit(iris.data)
#verificando os centroides do algoritmo
centroides = modelo.cluster_centers_
#previsoes 
previsoes = modelo.labels_
#dimensoes das previsoes
classe1, quantidade1 = np.unique(previsoes, return_counts = True)
#visualizando a marix de confusao
confusao = confusion_matrix(iris.target, previsoes)
#grafico do modelo
plt.scatter(iris.data[previsoes == 0, 0],
            iris.data[previsoes == 0, 1],
            c = 'green',
            label = 'Setosa')
plt.scatter(iris.data[previsoes == 1, 0],
            iris.data[previsoes == 1, 1],
            c = 'red',
            label = 'Versicolor')
plt.scatter(iris.data[previsoes == 2, 0],
            iris.data[previsoes == 2, 1],
            c = 'blue',
            label = 'Virginica')
plt.legend()
