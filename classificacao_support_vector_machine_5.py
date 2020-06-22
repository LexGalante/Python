# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:14:55 2019

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
from sklearn.model_selection import GridSearchCV


X, y = load_svmlight_file("C:/Users/Alex/Desktop/Projetos/Python/dados/australian.txt")
#acessando os primeiros dados, ncessário acessar o array de dados
X.A[:2,:]
#sepanrando base de treino e teste
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=random_state)
#criando o modelo
svc = svm.SVC(kernel='rbf', C = 2888, gamma = 0.000001).fit(X_train,y_train)
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000, 10000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]},
    {'kernel': ['poly'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000, 10000]}
]
#criando o grid search para buscaros melhores parametros de CONSTANTE e GAMMA
clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
#visualiando os melhores parametros possiveis
clf.best_params_

#realizando o deploy
joblib.dump(svc, 'C:/Users/Alex/Desktop/Projetos/Python/deploy/australian.joblib')
