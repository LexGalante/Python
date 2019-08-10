# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:37:35 2019

@author: Alex
"""
#importes necessários
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils import plot_model
#separação da base de estudos
base = datasets.load_iris()
previsores = base.data
classe = base.target
#preparação da classe
classe_dummy = np_utils.to_categorical(classe)
#separação da base de teste e treinamento
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe_dummy,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
#criando modelo
modelo = Sequential()
modelo.add(Dense(units = 5,#quantidade de neuronios da camada de escondida
                 input_dim = 4))#quantidade de nueronios na camada de entrada
modelo.add(Dense(units = 4))
modelo.add(Dense(units = 3, activation='softmax'))
#detalhes da rede neural
modelo.summary()
#treinando o modelo
modelo.compile(optimizer='adam',#algoritmo que faz o ajustes dos pesos minimizando os erros
               loss='categorical_crossentropy',
               metrics=['accuracy'])
modelo.fit(X_treinamento,
           y_treinamento,
           epochs=1000,#numero de vezes que o algoritmo ira ajustar os erros
           validation_data=(X_teste, y_teste))
#criando a predicao
previsoes = modelo.predict(X_teste)
previsoes = (previsoes > 0.5)
#visualizando a matriz de confusao
y_teste_matriz = [np.argmax(t) for t in y_teste]
y_previsao_matrix = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_teste_matriz, y_previsao_matrix)



