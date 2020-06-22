# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:59:53 2019

@author: Alex
"""
import pickle
from mineracao_texto2 import model

open("training_data", "wb")
data = pickle.load(open( "training_data", "rb" ))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# importar a conversação
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Carrega o modelo de rede neural obtido
model.load('./model.tflearn')

#Função para limpar os enunciados do usuário
def clean_up_sentence(sentence):
    # tokenize
    sentence_words = nltk.word_tokenize(sentence)
    # stem
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
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
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
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
