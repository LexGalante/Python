# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:47:24 2019

@author: Alex
"""
import nltk
from nltk.stem.lancaster import LancasterStemmer
from unicodedata import normalize
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

#instanciando o stemmer
stemer = LancasterStemmer()

texto = 'Mais de 20 toneladas de óleo foram recolhidas, nesta sexta-feira (18)'\
        ' em seis praias de três cidades do Litoral Sul de Pernambuco.' \
        'A informação foi repassada à noite, pelo secretário de Meio '\
        'Ambiente e Sustentabilidade do estado, José Bertotti. O número'\
        'é 1.566% maior que o volume de petróleo recolhido na quinta-feira'\
        '(17), quando 1,2 tonelada foi retirada do mar, no estado.'
        
texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
print(texto)
#separando os token
tokens = nltk.word_tokenize(texto)
print(tokens)
#seperando dados em stem
stems = [stemer.stem(w.lower()) for w in tokens]
print(stems)
#json com as intenções
with open('C:\\Users\\Alex\\Desktop\\Projetos\\Python\\dados\\intents.json') as json_data:
    intents = json.load(json_data)    
words = [] #Para armazenar as palavras do corpus/conversação
classes = [] #Para armazenar as classes (tags) do corpus/conversação
documents = [] #Para armazenar os documentos do corpus/conversação
stopwords = ['?','(',')',',']
#Processar cada intent da conversacao
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # extrair os tokes de cada sentença
        w = nltk.word_tokenize(pattern)
        # adicionar os tokens na lista words
        words.extend(w)
        # adicionar a palavra e a tag na lista documents
        documents.append((w, intent['tag']))
        # adicionar as tags na lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])            
#Extrair os radicais (stems) + converter as palavras para minúsculas
words = [stemer.stem(w.lower()) for w in words if w not in stopwords]
words = sorted(list(set(words)))

# Remover as duplicatas das classes
classes = sorted(list(set(classes)))

#Estruturas criadas até aqui (remover este trecho de código posteriormente)
print(len(documents), "documentos")
print(documents)
print(len(classes), "classes", classes)
print(len(words), "radicias únicos", words)


# Conjunto de dados para treinamento, matriz esparsa de palavras (bag of words) para cada sentença
# estruturas para treinamento
training = []
output = []
# array vazio para a saída do processamento
output_empty = [0] * len(classes)
# montar as listas para conjundo de dados de treinamento
for doc in documents:
    # criar a [bag of words]
    bag = []  # lista de palavras (tokems) de cada documento - serão utilizadas para obter os padrões da conversação
    # reveja a estrutura de 'documents'. O elemento 0 contem as palvras e o elemento 1 contem a classe / tag
    pattern_words = doc[0]
    # stem para cada palavra
    pattern_words = [stemer.stem(word.lower()) for word in pattern_words]
    # carregar os tokens na [bag of words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # print(bag)
    # output será '0' para cad tag e '1' para a tag atual (vetor esparso)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    # training recebe a [bag of words] e a linha de saída de cada documento
# print(training)

# embaralhar os dados para treinamento apra retirar vícios por conta da ordenação
# converter em um vetor numérico [np.array], que é a entrada das redes neurais
random.shuffle(training)
training = np.array(training)

# separar atributos (train_x) e classes (train_y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print('x', train_x[0])
print('y', train_y[0])

x: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
y: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# resetar dados remanescentes no grafo do tensor
tf.reset_default_graph()
# Construir a rede neural (com dois níveis)
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8) # Primeiro nível
net = tflearn.fully_connected(net, 8) # Segundo nível
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax') #Carrega os dados na rede neural
net = tflearn.regression(net) # Finaliza a rede neural

# Determina o tipo de rede neural (DNN) e configura o tensorflow (tensorboard)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Inicia o treinamento
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True) #Obtem o modelo
model.save('model.tflearn') # Salva o modelo para uso posterior

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

#open("training_data", "wb")
data = pickle.load(open( "training_data", "rb" ))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# importar a conversação
with open('dados/intents.json') as json_data:
    intents = json.load(json_data)

# Carrega o modelo de rede neural obtido
model.load('./model.tflearn')

#Função para limpar os enunciados do usuário
def clean_up_sentence(sentence):
    # tokenize
    sentence_words = nltk.word_tokenize(sentence)
    # stem
    sentence_words = [stemer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# =============================================

# Retorna a [bag of words] vetor: 0 or 1 para cada palavra que existe no enunciado do usuário
def bow(sentence, words, show_details=False):
    # tokenize
    sentence_words = clean_up_sentence(sentence)
    # [bag of words]
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

#===================================================

#executa o classificados
def classify(sentence):
    # classifica / prevê os possíveis resultados para a sentença do usuário
    results = model.predict([bow(sentence, words)])[0]
    # filtra as previsões de acordo com o limiar
    results = [[i,r] for i,r in enumerate(results) if r > 0.25]
    # ordena pela 'força' da distribuição probabilística
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # retora a lista de intençao / tag e o score de distribuição probabilística
    return return_list
#===================================================
    
#Determina a resposta para o enunciado do usuário
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # Se houve algum retorno, então encontrar a tag com maior score
    if results:
        # enquanto houver resultados a serem processados
        while results:
            for i in intents['intents']:
                # encontra uma tag que coincida com o resultado retornado
                if i['tag'] == results[0][0]:
                    # escolhe, aleatoriamente, a resposta a ser exposta
                    return print(random.choice(i['responses']))

            results.pop(0)
#===================================================


print(response('Quantos cilindros?'))

print(response('Qual é o valor?'))
print(response('Tá muito caro?'))
print(response('Esse ai é caro né'))

print(response('Quantas portas?'))

print(response('Valeu'))

print(response('Chico neves'))







