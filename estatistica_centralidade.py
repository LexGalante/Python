import numpy as np
from scipy import stats

salarios = [12335.55, 456778.99, 356778.99, 53544.77]

np.mean(salarios)

np.median(salarios)

quartis = np.quantile(salarios, [0, 0.25, 0.5, 0.75, 1])

np.std(salarios, ddof = 1)

stats.describe(salarios)

