"""
Natanael Emilio da Costa
matricula 16.1.8298
"""

"""
Para plotar o grafico Erro x Tempo precisa ter instalado as bibliotecas abaixo
py -m pip install -U pip
py -m pip install -U matplotlib
py -m pip install -U scikit-learn
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from csv import reader
import pandas as pd
import random
import numpy as np
import pylab as plotagem
from perceptronSpam import Perceptron

"""A base de dados deverá ser dividida em duas: uma para treinamento e uma para teste, na proporção
de 60% e 40%, respectivamente. Os dados deverão ser escolhidos aleatoriamente para não gerar
conjuntos de dados viciados ou polarizados. """

#df = pd.read_csv("C:/Users/natan/OneDrive/Documents/GitHub/trabalhoPerceptron/spambase.txt", header = None) # Lê o arquivo sem o cabeçalho
df = pd.read_csv('spambase.txt', header = None) # Lê o arquivo sem o cabeçalho
baseAprendisagem = df.sample(frac=0.6, replace=True, random_state=1)    # seleciona 60% das linhas de forma randomica
baseTeste = df.sample(frac=0.4, replace=True, random_state=1)   # seleciona 40% das linhas de forma randomica

novoSamples = baseAprendisagem.iloc[:,0:57].values.tolist()     # estabelece base de treinamento em forma de lista ignorando o ultimo paramentro que é o identificador spam
novoExit = baseAprendisagem.iloc[:,-1].replace(0,-1).values.tolist() # estabelece base de exit para treinamento em forma de lista selecionando o ultimo paramentro que é o identificador spam

testeSamples = baseTeste.iloc[:,0:57].values.tolist()   # estabelece base de testes em forma de lista ignorando o ultimo paramentro que é o identificador spam
testeExit = baseTeste.iloc[:,-1].values.tolist() # estabelece base de exit para testes em forma de lista selecionando o ultimo paramentro que é o identificador spam

# Inicializa o Perceptron
network = Perceptron(sample=novoSamples, exit = novoExit, learn_rate=0.50, epoch_number=400, bias=-1)

# Chamada ao treinamento
resultadoTreinamento = network.trannig()

# realizando os testes para todos os elementos do banco de testes, montando o vetor de predição
resultado = [ network.sort(item) for item in testeSamples ]

"""
Você deverá treinar o seu Perceptron com o conjunto de dados de treinamento e, posteriormente
testá-lo e avaliar sua performance por meio da matriz de confusão e das medidas de sensibilidade e
especifcidade, conforma já visto em aulas anteriores.
"""

valoresReais    = testeExit # pega os valores reais para analisar
valoresPreditos = resultado # pega os valores peditos para analisar

matrizConfusao = confusion_matrix(testeExit,resultado) #gera a matriz de confusão
acuracia = accuracy_score(testeExit,resultado) # captura a acuracia da matriz 

vp = matrizConfusao[0][0]
vn = matrizConfusao[1][1]
fp = matrizConfusao[1][0]
fn = matrizConfusao[0][1]

sensibilidade = vp / (vp + fn) # calcula sensibilidade
especificidade = vn / (vn + fp) # calcula especificidade

print("sensibilidade = ", round(sensibilidade,4))
print("acuracia = ", round(acuracia,4))
print("especificidade = ", round(especificidade,4))
print("matriz de confusão = \n", matrizConfusao) # imprime a matriz
print("matriz de confusão = ",classification_report(testeExit,resultado)) # imprime detalhes da matriz de confusão


""". Plote o gráfco com o histórico do Erro x Tempo (épocas)."""

plotagem.plot(resultadoTreinamento[1], resultadoTreinamento[0])
plotagem.xlabel('Tempo(Epocas)')
plotagem.ylabel('Erros')
plotagem.ylim(0, max(resultadoTreinamento[1]))
figura = plotagem.gcf()
plotagem.show()
figura.savefig('grafico_Erro_x_Epoca.png', format='png')