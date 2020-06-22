# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:39:02 2019

@author: Alex
"""
from sklearn import svm, datasets
from sklearn.externals import joblib

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
C = 1.0
svc = svm.SVC(kernel='linear', C=C,gamma='auto').fit(X, y)

joblib.dump(svc, 'iris.joblib')
svc_load = joblib.load('iris.joblib')

svc_load.predict(X)
