from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

norm.cdf(6, 8, 2)

norm.sf(6, 8, 2)

1 - norm.sf(6, 8, 2)

norm.cdf(6, 8, 2) + 1 - norm.sf(6, 8, 2)

norm.cdf(10, 8, 2) - norm.cdf(8, 8, 2)

dados = norm.rvs(size = 1000)

stats.probplot(dados, plot = plt)

stats.shapiro(dados)