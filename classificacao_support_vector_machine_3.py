# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:11:20 2019

@author: Alex
"""
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib

X, y = load_svmlight_file("C:/Users/Alex/Desktop/Projetos/Python/dados/australian.txt")
#acessando os primeiros dados, ncessário acessar o array de dados
X.A[:2,:]
#sepanrando base de treino e teste
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=random_state)
#criando o modelo
svc = svm.SVC(kernel='rbf', C = 2888, gamma = 0.000001).fit(X_train,y_train)
result_train = svc.predict(X_train)
print(accuracy_score(result_train, y_train))
result_test = svc.predict(X_test)
print(accuracy_score(result_test, y_test))
#realizando o deploy
joblib.dump(svc, 'C:/Users/Alex/Desktop/Projetos/Python/deploy/australian.joblib')