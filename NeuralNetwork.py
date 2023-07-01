import numpy as np

class NeuralNetwork:
    def __init__(self, tam_entrada, tam_interm, tam_saida):
        self.tam_entrada = tam_entrada
        self.tam_interm = tam_interm
        self.tam_saida = tam_saida
        self.weights1 = np.random.randn(self.tam_entrada, self.tam_interm)
        self.weights2 = np.random.randn(self.tam_interm, self.tam_saida)
        self.biases1 = np.zeros((1, self.tam_interm))
        self.biases2 = np.zeros((1, self.tam_saida))
    
    def forward(self, X):
        self.hidden_layer = np.dot(X, self.weights1) + self.biases1
        self.hidden_activation = self.sigmoid(self.hidden_layer)
        self.output_layer = np.dot(self.hidden_activation, self.weights2) + self.biases2
        self.output_activation = self.sigmoid(self.output_layer)
        return self.output_activation
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Usage example
tam_entrada = 2
tam_interm = 3
tam_saida = 1

nn = NeuralNetwork(tam_entrada, tam_interm, tam_saida)
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = nn.forward(input_data)
print(output_data)
