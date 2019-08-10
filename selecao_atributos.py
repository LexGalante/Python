# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:50:40 2019

@author: Alex
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

#carregamento dos dados
credito = pd.read_csv("Credit.csv")
#separando dados de previsao
previsores = credito.iloc[:,0:20].values
#separando a classe
classe = credito.iloc[:,20].values
#preparando dados discretos para criar o modelo
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:, 6])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder.fit_transform(previsores[:, 9])
previsores[:, 11] = labelencoder.fit_transform(previsores[:, 11])
previsores[:, 13] = labelencoder.fit_transform(previsores[:, 14])
previsores[:, 14] = labelencoder.fit_transform(previsores[:, 14])
previsores[:, 16] = labelencoder.fit_transform(previsores[:, 16])
previsores[:, 18] = labelencoder.fit_transform(previsores[:, 18])
previsores[:, 19] = labelencoder.fit_transform(previsores[:, 19])
#criando o modelo
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
#criando modelo de suporte a vetores
modelo_svm = SVC()
modelo_svm.fit(X_treinamento, y_treinamento)
#criando previsoes
previsoes = modelo_svm.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)
erro = 1 - accuracy_score(y_teste, previsoes)
#criando modelo de floresta randomica
modelo_randomForest = ExtraTreesClassifier()
modelo_randomForest.fit(X_treinamento, y_treinamento)
#visualizando os atributos mais importantes conforme coeficiente de determinação
importancias = modelo_randomForest.feature_importances_
#preparando base com atributos mais relevantes
X_treinamento_2 = X_treinamento[:, [0,1,2,3]]
X_teste_2 = X_teste[:, [0,1,2,3]]
#treinando novo modelo com base de atributos mais importantes
modelo_svm.fit(X_treinamento_2, y_treinamento)
previsoes = modelo_svm.predict(X_teste_2)
acuracia = accuracy_score(y_teste, previsoes)
erro = 1 - accuracy_score(y_teste, previsoes)








