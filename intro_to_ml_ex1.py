import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Part 1: Single-Layer Neural Network with Gradient Descent
class SingleLayerNN:
    def __init__(self, input_size, step_size=0.001):
        # Initialize the neural network with input size and step size
        self.step_size = step_size
        self.train_loss = []
        self.test_loss = []

    def sigmoid(self, z):
        # Sigmoid activation function
        z = np.clip(z, -700, 700)
        return 1 / (1 + np.exp(-z))

    def gradient_sigmoid(self, z):
        # Derivative of the sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cross_entropy_loss(self, y_true, y_pred):
        # Cross-entropy loss function
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def training(self, X_train, y_train, X_test, y_test, epochs=500):
        # Training function
        X_train_with_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        X_test_with_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
        w_0 = np.random.rand(X_train_with_bias.shape[1], len(X_test_with_bias[1])) * 0.01
        w_1 = np.random.rand(len(X_test_with_bias[1]), 1) * 0.01

        for epoch in range(epochs):
            #  gradient descent for each epoch
            w_0, w_1 = self.gradient_descent(X_train_with_bias, y_train, w_0, w_1, self.step_size)
            train_pred = self.predict(X_train, w_0, w_1)
            test_pred = self.predict(X_test, w_0, w_1)
            train_loss = np.mean(self.cross_entropy_loss(y_train, train_pred))
            test_loss = np.mean(self.cross_entropy_loss(y_test, test_pred))
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)

        # Store trained weights
        self.weights_0 = w_0
        self.weights_1 = w_1

    def gradient_descent(self, a_0, y, w_0, w_1, eta):
        # Gradient descent algorithm to update weights
        w_0_change = np.zeros_like(w_0)
        w_1_change = np.zeros_like(w_1)
        for i in range(a_0.shape[0]):
            a_1 = np.dot(a_0[i], w_0)
            z_1 = self.sigmoid(a_1)
            a_2 = np.dot(z_1, w_1)
            y_predicted = self.sigmoid(a_2)
            delta_2 = y_predicted - y[i]
            delta_1 = np.dot(delta_2, w_1.T) * self.gradient_sigmoid(a_1)
            grad_w_1 = np.dot(z_1[:, np.newaxis], delta_2[np.newaxis, :])
            grad_w_0 = np.dot(a_0[i][:, np.newaxis], delta_1[np.newaxis, :])
            w_0_change += grad_w_0
            w_1_change += grad_w_1
        w_0 -= eta * w_0_change
        w_1 -= eta * w_1_change
        return w_0, w_1

    def predict(self, X, w_0_trained, w_1_trained):
        # Predict function to make predictions with trained weights
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        a_1 = np.dot(X_with_bias, w_0_trained)
        z_1 = self.sigmoid(a_1)
        a_2 = np.dot(z_1, w_1_trained)
        y_predicted = self.sigmoid(a_2)
        return y_predicted

    def accuracy(self, X, y, w_0_trained, w_1_trained):
        # Calculate accuracy
        y_predicted = self.predict(X, w_0_trained, w_1_trained)
        y_predicted = np.round(y_predicted)
        return np.sum(y_predicted == y) / len(y)

    def plot_loss(self):
        # Plot training and testing loss over epochs
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

# Load data
data = load_breast_cancer()
X = data.data[:, 0:]  # Remove the first column and take the rest as X
y = data.target   # Take the second column as y

# Split data into 80/20 train/test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
split_index = int(0.8 * X.shape[0])
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train and evaluate the model
nn = SingleLayerNN(input_size=X_train.shape[1])
nn.training(X_train, y_train, X_test, y_test)
accuracy = nn.accuracy(X_test, y_test, nn.weights_0, nn.weights_1)
print(f'Accuracy: {accuracy}')
nn.plot_loss()
