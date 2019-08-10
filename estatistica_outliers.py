# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:49:23 2019

@author: Alex
"""

import matplotlib.pyplot as plt
import pandas as pd
from pyod.models.knn import KNN

#carregando dados
iris = pd.read_csv('iris.csv')
#verificando outliers atraves de boxlplot
plt.boxplot(iris.iloc[:, 1])
#extraindo outliers
outliers = iris[iris['sepal width'] > 4 | iris['sepal width'] < 2.1]
#utilizando a lib pyod 
sepal_width = iris.iloc[:, 1]
sepal_width = sepal_width.reshape(-1, 1)
detect_outliers = KNN()
detect_outliers.fit(sepal_width)
