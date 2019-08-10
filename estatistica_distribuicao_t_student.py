from scipy.stats import t
import numpy as np

# Media 75
# Amostra 9
# Desvio padrao 10
# T 1.5
# Probrabilidade de achar algo menor que 75 
probabilidade_menor_75 = t.cdf(1.5, 8)
# Probrabilidade de achar algo maior que 75
probabilidade_maior_75 = t.sf(1.5, 8)
# Checando resultado
probabilidade_total = np.round((probabilidade_menor_75 + probabilidade_maior_75) * 100)


teste = np.random(a = [0, 1], size = [50], replace = True, p = [0.5, 0.5, 0.2])