# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:25:23 2019

@author: Alex
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
#criando o modelo
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_treinamento, y_treinamento)
#acuracia
previsoes = modelo.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
acuracia = accuracy_score(y_teste, previsoes)
erro = 1 - accuracy_score(y_teste, previsoes)
#visualizando as arvores criadas
modelo.estimators_












