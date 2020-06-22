# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:01:03 2020

@author: Alex
"""

# padrão de de importação
import matplotlib.pyplot as plt
import numpy as np
from random import sample

# exemplo basico
x = np.linspace(0, 5, 11)
y = x*x

plt.plot(x, y, color='r')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Exemplo de PLOT')

# exemplo com multiplos graicos
plt.subplot(1, 2, 1)
plt.plot(x, y, 'g--')
plt.subplot(1, 2, 2)
plt.plot(x, y, 'r--')

# construção de um plot
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y)
axes.set_xlabel('Eixo X')
axes.set_ylabel('Eixo Y')
axes.set_title('Titulo')

# subplots
fig, ax = plt.subplots()
ax.plot(x, x**3, 'r--')
ax.plot(x, x**4, 'y--')
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_title('Titulo')


fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x, x**3, 'r--')
    ax.plot(x, x**4, 'y--')
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_title('Titulo')

# customizações
fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
ax.plot(x, x**3, label='x ^ 3', linewidth=5, linestyle='-.')
ax.plot(x, x**4, 'y--', label='x ^ 4', marker = '+', alpha=0.5)
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_title('Titulo')
ax.legend()

# pontos nos valores
plt.scatter(x, y)

# histograma
dados = sample(range(1, 10000), 1000)
plt.hist(dados)

# boxplot
dados = [np.random.normal(0, x, 100) for x in range(1, 5)]
plt.boxplot(dados)


