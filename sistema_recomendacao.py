# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:39:48 2020

@author: Alex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


colum_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('dados/filmes.data', sep='\t', names=colum_names)
print(df.head())
print(df.info())

movie_title = pd.read_csv('dados/filmes_titulos.csv')

df = pd.merge(df, movie_title, on='item_id')

# analises exploratorias
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head(10))
print(df.groupby('title')['rating'].count().sort_values(ascending=False).head(10))

rating = pd.DataFrame(df.groupby('title')['rating'].mean())
rating['count'] = pd.DataFrame(df.groupby('title')['rating'].mean())
rating['count'].hist()