import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# Part 1: Single-Layer Neural Network with Gradient Descent
class SingleLayerNN:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = np.zeros((1,))
        self.learning_rate = learning_rate


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def output(self, X):
        w_1= np.random.rand(len(X), len(X))
        z_1 = self.sigmoid(np.dot(X, w_1) + self.bias) # row len(m)
        w_2 = np.random.rand(len(X), 1)
        y_pred = self.sigmoid(np.dot(z_1, w_2) + self.bias) # scalar -> y_pred
        return y_pred

    def gradient_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred ) + (1 - y_true) * np.log(1 - y_pred))  #loss function


    def train(self, X, y, epochs=1000, tol=1e-4):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute loss
            loss = self.cross_entropy_loss(y, y_pred)
            losses.append(loss)

            # Compute gradients
            dz = y_pred - y
            dw = np.dot(X.T, dz) / len(y)
            db = np.sum(dz) / len(y)

            # Update weights and biases
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check for convergence
            if epoch > 0 and abs(losses[-1] - losses[-2]) < tol:
                print(f"Converged at epoch {epoch}")
                break

        return losses

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) > 0.5).astype(int)