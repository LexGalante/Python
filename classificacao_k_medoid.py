# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:42:32 2019

@author: Alex
"""
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
#definido o dataset
iris = datasets.load_iris()
#criando modelo
modelo = kmedoids(iris.data[:, 0:2], [3, 12, 20])
medoids = modelo.get_medoids()
#treinando o modelo
modelo.process()
#verificando resultador
previsoes = modelo.get_clusters()
#visualizando o clustes
view = cluster_visualizer()
view.append_clusters(previsoes, iris.data[:,0:2])
view.append_cluster(medoids, iris.data[:,0:2],
                    marker='*',
                    markersize = 15)
view.show();

        