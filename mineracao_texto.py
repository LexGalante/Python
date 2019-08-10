# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:22:26 2019

@author: Alex
"""
#instalando a lib nltk
#import nltk
#nltk.download()
import matplotlib.pyplot as plt
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from matplotlib.colors import ListedColormap
#pip install wordcloud (no anaconda prompt)
from wordcloud import WordCloud
#criando o corpus
corpus = PlaintextCorpusReader("C:/Users/Alex/Desktop/Projetos/Python/dados/mineracao_texto", ".*")
#importando os arquivos ids
arquivos = corpus.fileids()
#visualizando um arquivo especifico
arquivos[0:10]
#acessando o texto de um arquivo especifico
texto_de_um_arquivo = corpus.raw("1.txt")
texto_do_corpus = corpus.raw()
#visualizando palavras
palavras = corpus.words()
#palavras sem semantica
stops = stopwords.word('english')
#criando nuvem de palavras
nuvem = WordCloud(background_color='white',
                  colormap=ListedColormap(['orange', 'green', 'red', 'magenta']),
                  stopwords=stopwords.word('english'),
                  max_words=100)
nuvem.generate(texto_do_corpus)
plt.imshow(nuvem)
#visualizando os termos mais frequentes
palavras_stop_word = [p for p in palavras if p not in stops]
frequencia = nltk.FreDist(palavras_stop_word)
mais_comuns = frequencia.most_common(100)


