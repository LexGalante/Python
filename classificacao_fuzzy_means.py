# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:22:27 2019

@author: Alex
"""
from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import skfuzzy
#definido o dataset
iris = datasets.load_iris()
#criando o modelo
modelo = skfuzzy.cmeans(data = iris.data.T,#necessário transformar coluna em linha linha em coluna
                        c = 3,#numero de clusters
                        m = 2,#ver documentação
                        error = 0.005,#ver documentação
                        maxiter = 1000,#ver documentação
                        init = None)
#percentual das proporcoes
previsoes = modelo[1]
#probabilidade de o primeiro registro ser do cluester 0
np.round(previsoes[0][0] * 100, 2)
#probabilidade de o primeiro registro ser do cluester 1
np.round(previsoes[1][0] * 100, 2)
#probabilidade de o primeiro registro ser do cluester 2
np.round(previsoes[2][0] * 100, 2)
#matriz de confusao
previsao = previsoes.argmax(axis = 0)
confusao = confusion_matrix(iris.target, previsao)