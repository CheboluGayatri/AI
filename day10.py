import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error (MSE) Loss Function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Basic Neural Network
class BasicNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(1, output_size)

    # Forward pass
    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    # Backward pass
    def backward(self, x, y):
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += x.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    # Training function
    def train(self, x, y, epochs, learning_rate):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y)
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Predict function
    def predict(self, x):
        x = x.reshape(1, -1)  # Ensure correct shape
        return self.forward(x)

# XOR dataset
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize and train
nn = BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
nn.train(x, y, epochs=1000, learning_rate=0.1)

# Testing
print("\nTesting the trained neural network:")
for i in range(len(x)):
    output = nn.predict(x[i])
    print(f"Input: {x[i]}, Predicted Output: {output.round(3)}, Actual Output: {y[i]}")
