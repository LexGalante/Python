# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:34:47 2020

@author: Alex
"""

import numpy as np

vetor = [11,22,33]
matriz = [[100,200,300], [400,500,600], [700,800,900]]
# transoforma os objetos python em objetos numpy
# que encapsulam uma serie de funcionalidades adicionais
print(np.array(vetor))
print(np.array(matriz))
# gera um array de numeros de 0  a 100 de 2 em 2
print(np.arange(0, 100, 2))
# gera um np array de zeros
print(np.zeros(10))
# gera um np array de um
print(np.ones((3,3)))
# gera valores de uma distribuição normal
print(np.linspace(0,100,20))
# gera um array com numeros aleatorios
print(np.random.rand(5))
# gera uma matriz com numeros aleatorios
print(np.random.rand(5,5))
print(np.random.randn(5,5))
print(np.random.randint(0, 100, 5))
# transforma o array conforme dimensionalidade
vetor = np.random.rand(25)
vetor = vetor.reshape((5, 5))
# exibe a forma do array
print(vetor.shape)
# exibe o maior valor do array
print(vetor.max())
# exibe o menor valor do array
print(vetor.min())
# exibe a posição do maior valor
print(vetor.argmax())
# acessando valores, da mesma forma como se fosse em tuple, dict, set list
vetor = np.arange(0, 50, 5)
# acessando um item do array
print(vetor[3])
# acessando um range dentro do array, acessa o ultimo numero informado menos 1
print(vetor[2:6])
# se omitido considera 0
print(vetor[:3])
print(vetor[5:])
# acessando items em matrizez [LINHA INICIO : LINHA FIM , COLUNA INICIO : COLUNA FIM]
matriz = np.arange(50).reshape((5, 10))
print(matriz[:3])
print(matriz[:2, :1])
print(matriz[:1, :2])
# comparação
print(matriz > 45)
x = matriz > 45
matriz[x]
vetor_1 = np.arange(0, 20)
print(vetor_1.shape)
vetor_2 = np.arange(100, 120)
print(vetor_2.shape)
# calculos vetoriais
vetor_soma = vetor_1 + vetor_2
print(vetor_soma)
vetor_multiplicacao = vetor_1 * vetor_2
print(vetor_multiplicacao)
vetor_elevacao = vetor_1 ** vetor_2
print(vetor_elevacao)
print(vetor_1 + 2)
vetor_quadrado = np.sqrt(vetor_1)
print(vetor_quadrado)
vetor_exponencial = np.exp(vetor_1)
print(vetor_exponencial)
vetor_desvio_padrao = np.std(vetor_1)
print(vetor_desvio_padrao)
vetor_maximo = np.max(vetor_1)
print(vetor_maximo)
