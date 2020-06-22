# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:18:56 2020

@author: Alex
"""
import numpy as np
import pandas as pd

"""
SERIES
"""
labels = ['a', 'b',  'c']
lista = [10, 20, 30]
vetor = np.array(lista)
dicionario = {'a': 10, 'b': 20, 'c': 30}
# objeto Series é um dos principais da lib pandas, recebe qualquer tipo de dados
# tanto para chave como valor
serie = pd.Series(data=lista, index=labels)
print(serie['b'])
# operações aritmeticas entre series
# a operação acontece baseada em indice
serie_1 = pd.Series([1, 2, 3, 4], index=['BR', 'USA', 'ARG', 'ITA'])
serie_2 = pd.Series([1, 2, 3, 4], index=['BR', 'FRA', 'URU', 'ITA'])
print(serie_1 + serie_2)
"""
DATA FRAMES
"""
# setando o seed para numpy
np.random.seed(101)
df = pd.DataFrame(data=np.random.randn(5, 4),
                  index='A B C D E'.split(),
                  columns = 'W X Y Z'.split())
print(df)
# acessando dados em um dataframe
print(df['W'])
print(df[['W', 'Z']])
# criando colunas em uma dataframe
df['total'] = df['W'] + df['Z']
print(df)
# removendo colunas
df.drop('total', axis=1, inplace=True)
print(df)
# acessando dados pelo indice
print(df.loc['A', 'W'])
print(df.loc['A'])
print(df.loc[['A', 'B', 'E'], ['X', 'Z']])
# utilizando o ILOC acessamos dados como anotação de NUMPY
print(df.iloc[1:4, 2:])
# teste condicional
# dados maiore que zero
maior_zero = df > 0
print(maior_zero)
# redimensiona o dataframe conforme condicional
print(df[maior_zero])
print(df[df['W'] > 0]['Y'])
print(df[(df['W'] > 0) & (df['Y'] > 1)])
print(df[(df['W'] > 0) | (df['Y'] > 1)])
# resetando todos os indices
print(df.reset_index()) 
# removendo nulos
df.dropna()
# preenchendo os valores nulos
df.fillna(value=df['A'].mean())
# repete o valor nulo com o ultimo valor obtido
df.fillna(method='ffill')
# ordenando
dados = {'Empresa':['ABC', 'DEF', 'JJJ'],
         'Nome': ['FULANO', 'CICLANO', 'PAFUNCIO'],
         'Venda': [33354.99,48775.55, 34567.88]}
df = pd.DataFrame(dados)
print(df)
agrupamento = df.groupby('Empresa')
print(agrupamento.sum())
print(agrupamento.mean())
print(agrupamento.describe())
# concatenação
df_1 = pd.read_csv('dados/pandas_1.csv')
print(df_1)
df_2 = pd.read_csv('dados/pandas_2.csv')
print(df_2)
df_3 = pd.concat([df_1, df_2])
print(df_3)
print(pd.concat([df_1, df_2], axis=1))
# outras operações
print(df_1['cidade'].unique())
print(df_1['cidade'].value_counts())
print(df_1['curados'].apply(lambda x: x*x))
print(df_1.columns)
print(df_1.index)
df_1.sort_values(by='cidade', inplace=True)
print(df_1)
print(df_1.head())











