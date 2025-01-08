import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# Part 1: Single-Layer Neural Network with Gradient Descent
class SingleLayerNN:
    def __init__(self, input_size, step_size=0.01):
        self.step_size = step_size


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cross_entropy_loss(self, y_true, y_pred):
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred) #loss function

    def training(self, X, y, epochs = 1000): #X is NxM, y is 1x1 step_size is 0.01
        X_with_bias = np.stack(X, np.ones((X.shape[0], 1))) #X_with_bias is Nx(M+1)
        w_0 = np.random.rand(len(X_with_bias[0]), len(X_with_bias[0])) * 0.01 # weights + bias
        w_1 = np.random.rand(len(X_with_bias[0]), 1) # weights + bias change
        for epoch in range(epochs):
            w_0, w_1 = self.gradient_descent(X_with_bias, y, w_0, w_1, self.step_size)

        self.weights_1 = w_0
        self.weights_2 = w_1


    def gradient_descent(self, a_0, y, w_0, w_1, eta):
        w_0_change = np.zeros(len(a_0[0]), len(a_0[0])) * 0.01 # weights + bias change
        w_1_change = np.zeros(len(a_0[0]), 1) * 0.01 # weights + bias
        for i in range(a_0.shape[0]):
            a_1 =np.dot(a_0[i], w_0)
            z_1 = self.sigmoid(a_1)
            a_2 = np.dot(z_1, w_1)
            y_predicted = self.sigmoid(a_2)
            delta_2 = y_predicted - y
            delta_1 = np.dot(w_1, delta_2 * self.gradient_sigmoid(a_2))
            grad_w_1 = np.dot(a_1, delta_2)
            grad_w_0 = np.dot(a_0[i], delta_1)
            w_0_change += grad_w_0
            w_1_change += grad_w_1
        w_0 -= eta * w_0_change
        w_1 -= eta * w_1_change
        return w_0, w_1

    def predict(self, X):
        X_with_bias = np.stack(X, np.ones((X.shape[0], 1)))
        a_1 = np.dot(X_with_bias, self.weights_1)
        z_1 = self.sigmoid(a_1)
        a_2 = np.dot(z_1, self.weights_2)
        y_predicted = self.sigmoid(a_2)
        return y_predicted

    def accuracy(self, X, y):
        y_predicted = self.predict(X)
        y_predicted = np.round(y_predicted)
        return np.sum(y_predicted == y) / len(y)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()



