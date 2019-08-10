# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:04:47 2019

@author: Alex
"""
from scipy.stats import poisson

#em uma determinada rua ocorrem em média 2 acidentes de carro por dia
#calculo da probabilidade de ocorrer 3 acidentes
poisson.pmf(3, 2)
#calculo da probabilidade de ocorrer 3 ou menos
poisson.cdf(3, 2)
#calculo da probabilidade de ocorrer mais que 3
poisson.sf(3, 2)
