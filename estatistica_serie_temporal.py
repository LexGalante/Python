# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:53:40 2019

@author: Alex
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima

#preparando dados
base = pd.read_csv('AirPassengers.csv')
#transformando atributo em data
dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m')
#relizando a leitura novamento com o parse do dado
base = pd.read_csv('AirPassengers.csv',
                   parse_dates = ['Month'],
                   index_col = 'Month',
                   date_parser = dateparse)
#transformando o data frame em series
ts = base['#Passengers']
#visualizado dados
ts[1]
ts['1949-02']
ts[datetime(1949, 2, 1)]
ts['1950-01-01':'1950-07-31']
ts[:'1950-07-31']
ts['1950']
ts.index.max()
ts.index.min()
#visualizando grafico da serie
plt.plot(ts)
#agrupando por ano
ts_year = ts.resample('A').sum()
plt.plot(ts_year)
#agrupamento mensal
ts_month = ts.groupby([lambda x:x.month]).sum()
plt.plot(ts_month)
#agrupando por datas
ts_dates = ts['1960-01-01' : '1960-12-01']
plt.plot(ts_dates)
#deompondo a serie temporal
dec = seasonal_decompose(ts)

tendencia = dec.trend
plt.plot(tendencia)
sazional = dec.seasonal
plt.plot(sazional)
aleatorio = dec.resid
plt.plot(aleatorio)
#contruindo grafico 1
plt.subplot(4,1,1)
plt.plot(ts, label = 'Original')
plt.legend(loc = 'best')
#contruindo grafico 2
plt.subplot(4,1,2)
plt.plot(ts, label = 'Tendencia')
plt.legend(loc = 'best')
#contruindo grafico 3
plt.subplot(4,1,3)
plt.plot(ts, label = 'Sazionalidade')
plt.legend(loc = 'best')
#contruindo grafico 3
plt.subplot(4,1,4)
plt.plot(ts, label = 'Aleatoriedade')
plt.legend(loc = 'best')
#configurando o grafico
plt.tight_layout()
#previsao
media = ts.mean()
media_ultimo_ano = ts['1960-01-01': '1960-12-01'].mean()
#media movel
media_movel = ts.rolling(window = 12).mean()
#grafico da media movel
plt.plot(ts)
plt.plot(media_movel, color = 'red')
#criando previsao
previsao = []
for i in range(1, 13):
    superior = len(media_movel) - i
    inferior = superior - 1
    previsao.append(media_movel[inferior:superior].mean())
    
previsao = previsao[::-1]
plt.plot(previsao)
#implementando arima para calcular previsao
modelo = ARIMA(ts, order=(2, 1, 2))
modelo_treinado = modelo.fit()
modelo_treinado.summary()
previsao = modelo_treinado.forecast(steps = 12)[0]
eixo = ts.plot()
modelo_treinado.plot_predict('1960-01-01', '1962-01-01',
                             ax = eixo,
                             plot_insample = True)

