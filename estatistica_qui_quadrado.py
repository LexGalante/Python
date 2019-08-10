# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:21:14 2019

@author: Alex
"""
from scipy.stats import chi2_contingency
import numpy as np

#dados
assiste_novela = np.array([[19,6], [43,32]])
#calculando qui quadrado
chi2_contingency(assiste_novela)

