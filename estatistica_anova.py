# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:13:11 2019

@author: Alex
"""
    
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

#leitura de dados
tratamento = pd.read_csv("anova.csv", sep = ";")
#grafico boxplot 
tratamento.boxplot(by = 'Remedio', grid = False)
#analise da variancia
an = ols("Horas ~ Remedio", data = tratamento).fit()
#criando modelo
an_model = sm.stats.anova_lm(an)
#analise da variancia complexa
anc = ols("Horas ~ Remedio * Sexo", data = tratamento).fit()
#criando modelo
anc_model = sm.stats.anova_lm(anc)
#verificando teste tukey
tukey = MultiComparison(tratamento["Horas"], tratamento["Remedio"])
tukey_result = tukey.tukeyhsd()
print(tukey_result)
#verificar a coluna reject se true é possivel recusar a hipotese nula
#visualizando grafico
tukey_result.plot_simultaneous()