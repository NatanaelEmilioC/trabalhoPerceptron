from csv import reader
import pandas as pd
import random
import numpy as np

baseDados = pd.read_csv("C:/Users/natan/OneDrive/Documents/GitHub/trabalhoPerceptron/spambase.txt", header = None).iloc[:,0:57]
novoSamples = baseDados.head(int(4600 * 0.6))
amostra = baseDados.tail(int(4600 * 0.4))

baseExit = pd.read_csv("C:/Users/natan/OneDrive/Documents/GitHub/trabalhoPerceptron/spambase.txt", header = None).iloc[:,-1]
novoExit = baseExit.head(int(4600 * 0.6))
#gerar matriz
#listaAprendisagem = list(amostra.itertuples(index=False, name=None))
#listaTeste = list(teste.itertuples(index=False, name=None))
#print(novoExit)

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
            if erro == False:
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
            print('Classification: P1')
        else:
            print(('Sample: ', sample))
            print('Classification: P2')

# Funcao de Ativacao
    def sign(self, u):
        return 1 if u >= 0 else -1


samples = [
    [-0.6508, 0.1097, 4.0009],
    [-1.4492, 0.8896, 4.4005],
    [2.0850, 0.6876, 12.0710],
    [0.2626, 1.1476, 7.7985],
    [0.6418, 1.0234, 7.0427],
    [0.2569, 0.6730, 8.3265],
    [1.1155, 0.6043, 7.4446],
    [0.0914, 0.3399, 7.0677],
    [0.0121, 0.5256, 4.6316],
    [-0.0429, 0.4660, 5.4323],
    [0.4340, 0.6870, 8.2287],
    [0.2735, 1.0287, 7.1934],
    [0.4839, 0.4851, 7.4850],
    [0.4089, -0.1267, 5.5019],
    [1.4391, 0.1614, 8.5843],
    [-0.9115, -0.1973, 2.1962],
    [0.3654, 1.0475, 7.4858],
    [0.2144, 0.7515, 7.1699],
    [0.2013, 1.0014, 6.5489],
    [0.6483, 0.2183, 5.8991],
    [-0.1147, 0.2242, 7.2435],
    [-0.7970, 0.8795, 3.8762],
    [-1.0625, 0.6366, 2.4707],
    [0.5307, 0.1285, 5.6883],
    [-1.2200, 0.7777, 1.7252],
    [0.3957, 0.1076, 5.6623],
    [-0.1013, 0.5989, 7.1812],
    [2.4482, 0.9455, 11.2095],
    [2.0149, 0.6192, 10.9263],
    [0.2012, 0.2611, 5.4631]

]

exit = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1]

# Inicializa o Perceptron
#network = Perceptron(sample=samples, exit = exit, learn_rate=0.01, epoch_number=1000, bias=-1)
network = Perceptron(sample=novoSamples.values.tolist(), exit = novoExit.values.tolist(), learn_rate=0.01, epoch_number=1000, bias=-1)
# Chamada ao treinamento
network.trannig()
#print(type(samples))
#print(type(novoSamples.values.tolist()))
while True:
    sample = []
    for i in range(3):
        sample.insert(i, float(input('Valor: ')))
    network.sort(sample)