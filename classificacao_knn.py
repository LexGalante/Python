# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:15:41 2019

@author: Alex
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from scipy import stats

#carregando a base
iris = datasets.load_iris()
#visualização inicial
stats.describe(iris.data)

previsores = iris.data
classe = iris.target

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                        classe,
                                                                        test_size = 0.3,
                                                                        random_state = 0)
#treinando o modelo
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_treinamento, y_treinamento)
previsoes = knn.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
acuracia = accuracy_score(y_teste, previsoes)
erro = 1 - accuracy_score(y_teste, previsoes)


