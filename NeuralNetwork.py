import numpy as np

class NeuralNetwork:
    def __init__(self, tam_entrada, tam_interm, tam_saida):
        self.tam_entrada = tam_entrada
        self.tam_interm = tam_interm
        self.tam_saida = tam_saida
        self.weights_ih = np.random.randn(self.tam_interm, self.tam_entrada)
        self.weights_ho = np.random.randn(self.tam_saida, self.tam_interm)

    def forward(self, inputs):
        hidden = np.dot(self.weights_ih, inputs)
        hidden = self.sigmoid(hidden)
        output = np.dot(self.weights_ho, hidden)
        output = self.sigmoid(output)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class GeneticAlgorithm:
    def __init__(self, population_size, tam_entrada, tam_interm, tam_saida):
        self.population_size = population_size
        self.tam_entrada = tam_entrada
        self.tam_interm = tam_interm
        self.tam_saida = tam_saida
        self.population = []
        for _ in range(population_size):
            network = NeuralNetwork(tam_entrada, tam_interm, tam_saida)
            self.population.append(network)

    def evaluate_population(self, inputs, targets):
        for network in self.population:
            total_error = 0
            for i in range(len(inputs)):
                output = network.forward(inputs[i])
                error = np.mean((output - targets[i]) ** 2)
                total_error += error
            network.fitness = 1 / (total_error + 1)

    def select_parents(self):
        fitness_scores = [network.fitness for network in self.population]
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        parents = np.random.choice(self.population, size=2, p=probabilities)
        return parents

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(self.tam_entrada, self.tam_interm, self.tam_saida)
        child.weights_ih = np.copy(parent1.weights_ih)
        child.weights_ho = np.copy(parent2.weights_ho)
        return child

    def mutate(self, network):
        mutation_rate = 0.01
        for i in range(len(network.weights_ih)):
            for j in range(len(network.weights_ih[i])):
                if np.random.random() < mutation_rate:
                    network.weights_ih[i][j] += np.random.randn() * 0.1
        for i in range(len(network.weights_ho)):
            for j in range(len(network.weights_ho[i])):
                if np.random.random() < mutation_rate:
                    network.weights_ho[i][j] += np.random.randn() * 0.1

population_size = 50
tam_entrada = 2
tam_interm = 4
tam_saida = 1

# Create an instance of the GeneticAlgorithm class
ga = GeneticAlgorithm(population_size, tam_entrada, tam_interm, tam_saida)

# Evaluate the fitness of the population
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
ga.evaluate_population(inputs, targets)

# Select parents for reproduction
parents = ga.select_parents()

# Perform crossover to create offspring
child = ga.crossover(parents[0], parents[1])

# Apply mutation to the weights of offspring
ga.mutate(child)
