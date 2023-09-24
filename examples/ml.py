import numpy as np

class Neuron:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def forward(self, inputs):
        # Compute weighted sum and apply activation function (sigmoid)
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.sigmoid(weighted_sum)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs):
        # Compute forward pass for each neuron in the layer
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return outputs

def train_neural_network(X, y, learning_rate, epochs):
    input_size = X.shape[1]
    output_size = y.shape[1]

    layer = NeuralLayer(input_size, output_size)

    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward pass
            inputs = X[i]
            targets = y[i]
            outputs = layer.forward(inputs)

            # Compute loss (mean squared error)
            loss = np.mean((targets - outputs) ** 2)

            # Backpropagation
            for j in range(output_size):
                error = targets[j] - outputs[j]
                gradient = error * outputs[j] * (1 - outputs[j])  # Derivative of sigmoid
                for k in range(input_size):
                    layer.neurons[j].weights[k] += learning_rate * gradient * inputs[k]
                layer.neurons[j].bias += learning_rate * gradient

            print(f"Epoch {epoch + 1}, Sample {i + 1}, Loss: {loss:.4f}")

# Sample training datasets (X) and targets (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Hyperparameters
learning_rate = 0.5
epochs = 10

# Train the neural network
train_neural_network(X, y, learning_rate, epochs)
