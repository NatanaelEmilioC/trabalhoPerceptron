"""
Natanael Emilio da Costa
matricula 16.1.8298
"""

from csv import reader
import pandas as pd
import random
import numpy as np

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

        listaDeErros = [] # lista de quantidade de Erros na etapa de treinamento
        listaDeEpocas = [] # lista para armazenar as epocas
        valoresObtidos = [] # lista respostas obtidas na epoca - usado para calcular o numero de erros

        #Metodo do Gradiente Descendente para ajuste dos pesos do Perceptron
        while True:
            
            valoresObtidos = []
            erro = False
            criterioDeParada = False

            for i in range(self.number_sample):
                u = 0
                for j in range(self.col_sample + 1):
                    u = u + self.weight[j] * self.sample[i][j]
                y = self.sign(u)
                if y != self.exit[i]:
                    for j in range(self.col_sample + 1):
                        self.weight[j] = self.weight[j] + self.learn_rate * (self.exit[i] - y) * self.sample[i][j]
                    erro = True
                valoresObtidos.append(y) # adiciona a resposta à lista de resposta na epoca

            print('Epoca: \n',epoch_count)
            
            listaDeEpocas.append(epoch_count) # armazena a epoca atual no vetor de epocas

            epoch_count = epoch_count + 1 # incrementa o indicador da epoca

            quantidadeDeErros = sum((1) for a, b in zip(valoresObtidos, self.exit) if a!=b) #calcular a quantidade de erros na etapa
            listaDeErros.append(quantidadeDeErros) # adiciona o numero de erros à lista
            
            if epoch_count > 1000: # estabelece uma epoca minima para não quebra a comparação com os ultimos erros
                nUltimos = listaDeErros[-1000 : -1] # recupera os 199 penultimos erros
                resultado = [numero for numero in nUltimos if numero < listaDeErros[-1]] # compara o ultimo valor de erros com os 199 valores anteriores
            
                if len(resultado) == len(nUltimos) : # caso o ultimo valor de erros seja maior que os outros 199 o tamando das listas será igual
                    criterioDeParada = True # atribui a condição para parada
                else:
                    criterioDeParada = False

            """
            Essa base de dados possui uma sobreposição espacial dos dados e por isso, o Perceptron não
            conseguirá separar perfeitamente os dados por meio de uma superfície linear de separação.

            Para isso, defna um critério de parada do algoritmo de treinamento que avalie o erro ao longo do
            treinamento e até um momento que ele não mais diminua
            """
            # parada por erro ou criterio de parada(os erros pararam de diminuir pelo menos nos ultimos 200)
            if erro == False or criterioDeParada == True:
                print(('\nEpocas:\n',epoch_count))
                print('------------------------\n')
                break
        
        return(listaDeErros, listaDeEpocas) # retorna a lista de erros e lista de epocas para montar o grafico

    def sort(self, sample):
        sample.insert(0, self.bias)
        u = 0
        for i in range(self.col_sample + 1):
            u = u + self.weight[i] * sample[i]

        y = self.sign(u)

        if  y == -1:
            #print('Classification: non-spam')
            return 0 # retorna para montar o matriz de confusão
        else:
            #print('Classification: spam')
            return 1 # retorna para montar a matriz de confusão

    # Funcao de Ativacao
    def sign(self, u):
        return 1 if u >= 0 else -1