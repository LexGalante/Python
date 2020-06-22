# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:51:41 2019

@author: Alex
"""
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#criando o modelo
X, y = load_svmlight_file("C:/Users/Alex/Desktop/Projetos/Python/dados/a1a.txt")
random_state = np.random.RandomState(0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.5, random_state=random_state)
#configuracoes para cada kernel
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



