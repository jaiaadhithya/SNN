import numpy as np
import itertools

# -------------------------------
# Activation functions
# -------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -------------------------------
# Generate network architectures
# -------------------------------
def generate_architectures(input_neurons, output_neurons, hidden_layer_sizes, num_hidden_layers):
    for combo in itertools.product(hidden_layer_sizes, repeat=num_hidden_layers):
        yield (input_neurons, *combo, output_neurons)

# -------------------------------
# Feedforward neural network
# -------------------------------
class FeedforwardNN:
    def __init__(self, layers):
        self.layers = layers
        # Initialize weights randomly between -1 and 1
        self.weights = [np.random.uniform(-1, 1, (layers[i+1], layers[i])) for i in range(len(layers)-1)]

    def forward(self, X):
        activations = [X]
        for w in self.weights:
            X = sigmoid(np.dot(w, X))
            activations.append(X)
        return activations

    def backward(self, X, y, activations, lr=0.01):
        deltas = []
        error = activations[-1] - y
        delta = error * sigmoid_derivative(activations[-1])
        deltas.append(delta)
        # Backpropagation
        for i in reversed(range(len(self.weights)-1)):
            delta = np.dot(self.weights[i+1].T, deltas[-1]) * sigmoid_derivative(activations[i+1])
            deltas.append(delta)
        deltas.reverse()
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= lr * np.outer(deltas[i], activations[i])
        return np.mean(error**2)

# -------------------------------
# Training function
# -------------------------------
def train_and_get_error(architecture, X_train, y_train, epochs=200, lr=0.01):
    nn = FeedforwardNN(architecture)
    for epoch in range(epochs):
        error = 0
        for x, y in zip(X_train, y_train):
            activations = nn.forward(x)
            error += nn.backward(x, y, activations, lr)
        error /= len(X_train)
    return error, nn.weights

# -------------------------------
# Fixed dataset (9 inputs, 4 outputs)
# -------------------------------
X_train = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    [0.5, 0.1, 0.4, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9],
    [0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.9],
    [0.6, 0.4, 0.9, 0.1, 0.7, 0.3, 0.2, 0.5, 0.8],
    [0.2, 0.3, 0.1, 0.5, 0.6, 0.4, 0.7, 0.8, 0.9],
    [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.9],
    [0.1, 0.4, 0.2, 0.5, 0.3, 0.6, 0.7, 0.8, 0.9],
    [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5],
    [0.5, 0.2, 0.6, 0.3, 0.7, 0.1, 0.4, 0.8, 0.9],
    [0.3, 0.6, 0.1, 0.8, 0.2, 0.7, 0.4, 0.5, 0.9],
    [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.8, 0.9],
    [0.2, 0.5, 0.3, 0.6, 0.4, 0.7, 0.1, 0.8, 0.9],
    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9],
    [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8],
    [0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.4, 0.5],
    [0.4, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9],
    [0.3, 0.5, 0.2, 0.6, 0.1, 0.7, 0.4, 0.8, 0.9],
    [0.7, 0.1, 0.6, 0.2, 0.5, 0.3, 0.4, 0.8, 0.9],
    [0.2, 0.4, 0.1, 0.3, 0.5, 0.7, 0.6, 0.8, 0.9]
])

y_train = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.4, 0.3, 0.2, 0.1],
    [0.5, 0.6, 0.7, 0.8],
    [0.8, 0.7, 0.6, 0.5],
    [0.9, 0.1, 0.4, 0.2],
    [0.2, 0.3, 0.1, 0.5],
    [0.6, 0.5, 0.4, 0.3],
    [0.1, 0.4, 0.2, 0.5],
    [0.9, 0.2, 0.7, 0.3],
    [0.5, 0.3, 0.6, 0.2],
    [0.4, 0.6, 0.2, 0.8],
    [0.7, 0.5, 0.4, 0.6],
    [0.2, 0.5, 0.3, 0.7],
    [0.8, 0.7, 0.6, 0.5],
    [0.1, 0.3, 0.5, 0.7],
    [0.9, 0.2, 0.8, 0.1],
    [0.4, 0.1, 0.5, 0.2],
    [0.3, 0.5, 0.2, 0.6],
    [0.7, 0.1, 0.6, 0.2],
    [0.2, 0.4, 0.1, 0.3]
])

# -------------------------------
# Define architectures
# -------------------------------
hidden_layer_sizes = [12, 16, 20, 24]
num_hidden_layers = 3
input_neurons = 9    # matches X_train columns
output_neurons = 4   # matches y_train columns
architectures = list(generate_architectures(input_neurons, output_neurons, hidden_layer_sizes, num_hidden_layers))

# -------------------------------
# Train all architectures
# -------------------------------
results = []
for arch in architectures:
    err, weights = train_and_get_error(arch, X_train, y_train, epochs=100, lr=0.05)
    print(f"Architecture {arch} --> Final Error = {err:.4f}")
    results.append((arch, err, weights))

# -------------------------------
# Find best architecture
# -------------------------------
best_arch, best_err, best_weights = min(results, key=lambda x: x[1])
print("\n==============================================")
print(" BEST ARCHITECTURE BASED ON MINIMUM FINAL ERROR")
print("==============================================")
print(f"Architecture: {best_arch}")
print(f"Final Error: {best_err:.4f}")
print("\n--- Detailed Weight Assignment (Best Architecture Only) ---")
for i, w in enumerate(best_weights):
    print(f"Layer {i+1} Weights:\n{w}\n")
