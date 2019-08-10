# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:26:38 2019

@author: Alex
"""
#instalar o pacote pip install apyori
from apyori import apriori
import pandas as pd
#importando os dados
dados = pd.read_csv("transacoes.txt", header = None)
#transformando os dados para utilizar o apyori
transacoes = []
for i in range(0, 6):
    transacoes.append([str(dados.values[i,j]) for j in range(0,3)])
#extraindo as regras
regras = apriori(transacoes,min_support=0.5,min_confidenc=0.5)
#visualizando regras extraidas
resultados = list(regras)
resultados_split = [list(x) for x in resultados]
resultador_parse = []
for j in range(0,7):
    resultador_parse.append([list(x) for x in resultados_split[j][2]])
