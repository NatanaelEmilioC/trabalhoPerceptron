from csv import reader
import pandas as pd
import random
import numpy as np

df = pd.read_csv("C:/Users/natan/OneDrive/Documents/GitHub/trabalhoPerceptron/spambase.txt", header = None)
baseAprendisagem = df.sample(frac=0.6, replace=True, random_state=1)
baseTeste = df.sample(frac=0.4, replace=True, random_state=1)

novoSamples = baseAprendisagem.iloc[:,0:57].values.tolist()
novoExit = baseAprendisagem.iloc[:,-1].replace(0,-1).values.tolist()

testeSamples = baseTeste.iloc[:,0:57].values.tolist()
testeExit = baseTeste.iloc[:,-1].values.tolist()

print(novoExit)

class Perceptron:

    # Inicializacao do objeto Perceptron
    def __init__(self, sample, exit, learn_rate=0.01, epoch_number=1000, bias=-1):
        self.sample = sample
        self.exit = exit
        self.learn_rate = learn_rate
        self.epoch_number = epoch_number
        self.bias = bias
        self.number_sample = len(sample)
        self.col_sample = len(sample[0])
        self.weight = []

    # Funcao de Treinamento do Perceptron (Metodo Gradiente Descendente)
    def trannig(self):
        for sample in self.sample:
            sample.insert(0, self.bias)
        
        # Inicializa os pesos w aleatoriamente
        for i in range(self.col_sample):
           self.weight.append(random.random())

        # Insere peso da entrada de polarizacao(bias)
        self.weight.insert(0, self.bias)

        epoch_count = 0

        #Metodo do Gradiente Descendente para ajuste dos pesos do Perceptron
        while True:
            erro = False
            for i in range(self.number_sample):
                u = 0
                for j in range(self.col_sample + 1):
                    u = u + self.weight[j] * self.sample[i][j]
                y = self.sign(u)
                if y != self.exit[i]:
                    for j in range(self.col_sample + 1):
                        self.weight[j] = self.weight[j] + self.learn_rate * (self.exit[i] - y) * self.sample[i][j]
                    erro = True
            print('Epoca: \n',epoch_count)
            epoch_count = epoch_count + 1
            # Se parada porepocas ou erro
            if erro == False or epoch_count > 5000:
                print(('\nEpocas:\n',epoch_count))
                print('------------------------\n')
                break

    def sort(self, sample):
        sample.insert(0, self.bias)
        u = 0
        for i in range(self.col_sample + 1):
            u = u + self.weight[i] * sample[i]

        y = self.sign(u)

        if  y == -1:
            print(('Sample: ', sample))
            print('Classification: nospam')
        else:
            print(('Sample: ', sample))
            print('Classification: spam')

# Funcao de Ativacao
    def sign(self, u):
        return 1 if u >= 0 else -1


# Inicializa o Perceptron
network = Perceptron(sample=novoSamples, exit = novoExit, learn_rate=0.01, epoch_number=1000, bias=-1)

# Chamada ao treinamento
network.trannig()

network.sort(testeSamples[1])
print(testeExit[1])

network.sort(testeSamples[2])
print(testeExit[2])

network.sort(testeSamples[5])
print(testeExit[5])

network.sort(testeSamples[10])
print(testeExit[10])

"""
while True:
    sample = []
    for i in range(3):
        sample.insert(i, float(input('Valor: ')))
    network.sort(sample)
"""