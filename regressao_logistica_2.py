# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:44:42 2020

@author: Alex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


%matplotlib inline

df = pd.read_csv('dados/titanic_train.csv')
# informações basícas
print(df.head())
print(df.info())
# visualizando os dados nulos
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# passageiros que sobreviveram
sns.set_style('white')
# sexo com maior numero de sobreviventes
sns.countplot(x='Survived', data=df, hue='Sex')
# idade
df['Age'].hist()
# acompanhantes
sns.countplot(x='SibSp', data=df)
# distribuição do preço pago por passagem
df['Fare'].hist()
df[df['Fare'] < 70]['Fare'].hist()
# dados faltantes de idade
sns.boxplot(x='Pclass', y='Age', data=df)
# preenchendo as idades faltantes com a média por classe das idades 
# faltantes
def input_age(cols):
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age
        
def prepare_data(dataset):
    dataset['Age'] = dataset[['Age', 'Pclass']].apply(input_age, axis=1)
    # removendo a coluna cabine, pois nao há dados
    dataset.dropna()
    # transformanda dados categoricos em continuos
    sex = pd.get_dummies(dataset['Sex'], drop_first=True)
    embarked = pd.get_dummies(dataset['Embarked'], drop_first=True)
    dataset.drop(['PassengerId',
                  'Sex',
                  'Embarked',
                  'Name',
                  'Ticket',
                  'Cabin'],
                 axis=1,
                 inplace=True)
    dataset = pd.concat([dataset, sex, embarked], axis=1)
    
    return dataset

df = prepare_data(df)
x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1),
                                                    df['Survived'],
                                                    test_size=0.2)
# criando e treinando o modelo
model = LogisticRegression()
model.fit(x_train, y_train)
# predições
predicts = model.predict(x_test)
print(classification_report(y_test, predicts))
# matriz de confusão
print(confusion_matrix(y_test, predicts))
