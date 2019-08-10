# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:56:31 2019

@author: Alex
"""
#importes necessários
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.utils import plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
#import o data set mnist já separa para treinamento e teste
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
#visualizando um dos numeros
plt.imshow(y_treinamento[2], img = 'gray')
plt.title(y_treinamento[2])
#transformando os dados
X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))
X_treinamento = X_treinamento.astype('float32')
X_teste = X_teste.astype('float32')
X_treinamento /= 255
X_teste /= 255
y_treinamento = np_utils.to_categorical(y_treinamento)
y_teste = np_utils.to_categorical(y_teste)
#criando modelo
modelo = Sequential()
modelo.add(Dense(units=64, activation='relu', input_dim=784))
modelo.add(Dropout(0.2))
modelo.add(Dense(units=64, activation='relu', input_dim=784))
modelo.add(Dropout(0.2))
modelo.add(Dense(units=64, activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 10, activation='softmax'))
modelo.summary()
#compilando e gerando e treinando
modelo.compile(optimizer='adam',#algoritmo que faz o ajustes dos pesos minimizando os erros
               loss='categorical_crossentropy',
               metrics=['accuracy'])
historico = modelo.fit(X_treinamento,
                       y_treinamento,
                       epochs = 20,
                       validation_data=(X_teste, y_teste))
historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_acc'])
#realizando previsoes
previsoes = modelo.predict(X_teste)
#preparando a matriz de confusao
y_teste_matriz = [np.argmax(t) for t in y_teste]
y_previsao_matrix = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_teste_matriz, y_previsao_matrix)