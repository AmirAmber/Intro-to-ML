from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, step_size=0.01):
        self.step_size = step_size
        self.train_loss = []
        self.test_loss = []
        self.W_0 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b_0 = np.zeros(hidden_size)
        self.W_1 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b_1 = np.zeros(output_size)

    def sigmoid(self, z):
        z = np.clip(z, -700, 700)
        return 1 / (1 + np.exp(-z))

    def gradient_sigmoid(self, z):  # derivative of the activation function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def forward_pass(self, X):
        a_1 = np.dot(X, self.W_0) + self.b_0
        z_1 = self.sigmoid(a_1)
        a_2 = np.dot(z_1, self.W_1) + self.b_1
        z_2 = self.sigmoid(a_2)
        return a_1, z_1, a_2, z_2

    def backward_propagation(self, X, y):
        a_1, z_1, a_2, z_2 = self.forward_pass(X)
        m = X.shape[0]
        delta_2 = z_2 - y.reshape(-1, 1)
        dW_1 = np.dot(z_1.T, delta_2) / m
        db_1 = np.sum(delta_2) / m
        delta_1 = np.dot(delta_2, self.W_1.T) * self.gradient_sigmoid(a_2)
        dW_0 = np.dot(X.T, delta_1) / m
        db_0 = np.sum(delta_1) / m
        return dW_0, db_0, dW_1, db_1

    def gradient_descent(self, dW_0, db_0, dW_1, db_1):
        self.W_0 -= self.step_size * dW_0
        self.b_0 -= self.step_size * db_0
        self.W_1 -= self.step_size * dW_1
        self.b_1 -= self.step_size * db_1

    def nn_predict(self, X):
        a_1, z_1, a_2, z_2 = self.forward_pass(X)
        return z_2


    def nn_training(self, X_train, y_train, X_test, y_test, epochs=1000, tolerance=1e-4):
        for epoch in range(epochs):
            dW_0, db_0, dW_1, db_1 = self.backward_propagation(X_train, y_train)
            self.gradient_descent(dW_0, db_0, dW_1, db_1)

            train_pred = self.nn_predict(X_train)
            test_pred = self.nn_predict(X_test)

            train_loss_for_epoch = np.mean(self.cross_entropy_loss(y_train, train_pred))
            test_loss_for_epoch = np.mean(self.cross_entropy_loss(y_test, test_pred))

            self.train_loss.append(train_loss_for_epoch)
            self.test_loss.append(test_loss_for_epoch)

            if epoch > 0 and abs(self.train_loss[-2] - self.train_loss[-1]) < tolerance:
                print(f"Training stopped early at epoch {epoch} due to minimal loss change.")
                break

        self.weights = self.W_0, self.W_1
        self.biases = self.b_0, self.b_1


    def print_weights(self):
        print(f'W0 is: {self.W_0} and its bias is: {self.b_0}')
        print(f'W1 is: {self.W_1} and its bias is: {self.b_1}')

    def nn_accuracy(self, X, y):
        y_predicted = self.nn_predict(X)
        y_predicted = (y_predicted > 0.5).astype(int)
        accuracy = np.sum(y_predicted == y.reshape(-1, 1)) / len(y)
        print(f'The accuracy of this neural network is: {accuracy * 100:.2f}%')
        return accuracy

    def nn_plot_loss(self, accuracy):
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve (Accuracy: {accuracy * 100:.2f}%)')
        plt.show()

data = load_breast_cancer()
X = data.data  # features
y = data.target  # target variable

# Split data into 80/20 train/test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
split_index = int(0.8 * X.shape[0])
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize and train the network
nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=50, output_size=1, step_size=0.01)
nn.nn_training(X_train, y_train, X_test, y_test, epochs=800)
accuracy_of_nn = nn.nn_accuracy(X_test, y_test)
nn.print_weights()
nn.nn_plot_loss(accuracy_of_nn)