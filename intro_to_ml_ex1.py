import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Part 1: Single-Layer Neural Network with Gradient Descent
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

    def back_propagation(self, X, y):
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

    def predict_nn(self, X):
        a_1, z_1, a_2, z_2 = self.forward_pass(X)
        return z_2


    def training_nn(self, X_train, y_train, X_test, y_test, epochs=1000, tolerance=1e-4):
        for epoch in range(epochs):
            dW_0, db_0, dW_1, db_1 = self.back_propagation(X_train, y_train)
            self.gradient_descent(dW_0, db_0, dW_1, db_1)

            train_pred = self.predict_nn(X_train)
            test_pred = self.predict_nn(X_test)

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

    def accuracy_nn(self, X, y):
        y_predicted = self.predict_nn(X)
        y_predicted = (y_predicted > 0.5).astype(int)
        accuracy = np.sum(y_predicted == y.reshape(-1, 1)) / len(y)
        print(f'The accuracy of this neural network is: {accuracy * 100:.2f}%')
        return accuracy

    def plot_loss_nn(self, accuracy):
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve (Accuracy: {accuracy * 100:.2f}%)')
        plt.show()


# Part 2: decision tree with ID3 algorithm with entropy potential function

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        positive_examples = np.sum(y == 1) # Number of positive examples
        q = positive_examples / len(y)
        return -0.5*(q * np.log2(q) + (1-q)*np.log2(1-q)) if q > 0 and q < 1 else 0

    def information_gain(self, X_column, y, threshold):  # Calculate the information gain
        before_split_entropy = self.entropy(y)  # Calculate the entropy before the split
        left_indices = X_column <= threshold # Indices for the left side after the split
        right_indices = X_column > threshold  # Indices for the right side after the split
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0  # If either side is empty, return 0 gain
        n = len(y)
        n_left = np.sum(left_indices)
        n_right = np.sum(right_indices)
        e_left = self.entropy(y[left_indices])  # Calculate entropy for each side
        e_right = self.entropy(y[right_indices])
        after_split_entropy = (n_left / n) * e_left + (n_right / n) * e_right  # Weighted average entropy of after the split
        return before_split_entropy - after_split_entropy  # Information gain



    def best_split(self, X, y):
        best_gain = -1
        predicator = None
        best_threshold = None
        for p in range(X.shape[1]):
            thresholds = np.unique(X[:, p])
            for threshold in thresholds:
                gain = self.information_gain(X[:, p], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    predicator = p
                    best_threshold = threshold
        return predicator, best_threshold

    def build_tree(self, X, y, depth=0):
        if np.all(y == 1) or np.all(y == 0) or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()
        predicator, threshold = self.best_split(X, y)
        if predicator is None:
            return np.bincount(y).argmax()
        left_indices = X[:, predicator] <= threshold
        right_indices = X[:, predicator] > threshold
        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        return (predicator, threshold, left_subtree, right_subtree)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample_dt(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree
        predicator, threshold, left_subtree, right_subtree = tree
        if sample[predicator] <= threshold:
            return self.predict_sample_dt(sample, left_subtree)
        else:
            return self.predict_sample_dt(sample, right_subtree)

    def predict_dt(self, X):
        return np.array([self.predict_sample_dt(sample, self.tree) for sample in X])


    def accuracy_of_dt(self, X, y):
        predictions = self.predict_dt(X)
        accuracy = np.sum(predictions == y) / len(y)
        print(f'The accuracy of this decision tree is: {accuracy * 100:.2f}%')
        return accuracy

    def plot_tree(self):
        def plot_node(ax, tree, x, y, dx, dy):
            if not isinstance(tree, tuple):
                ax.text(x, y, str(tree), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
                return
            predicator, threshold, left_subtree, right_subtree = tree
            node_label = f"X[{predicator}]\n<=\n{threshold}"
            ax.text(x, y, node_label, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
            ax.plot([x, x - dx], [y - dy, y - 2 * dy], 'k-')
            ax.plot([x, x + dx], [y - dy, y - 2 * dy], 'k-')
            plot_node(ax, left_subtree, x - dx, y - 2 * dy, dx / 1.8, dy)
            plot_node(ax, right_subtree, x + dx, y - 2 * dy, dx / 1.8, dy)

        fig, ax = plt.subplots(figsize=(28, 28))
        ax.set_axis_off()
        plot_node(ax, self.tree, 1, 1, 16, 12)
        plt.show()
        ax.set_axis_off()
        plt.show()








# Load data
data = load_breast_cancer()
X = data.data # features
y = data.target   # target variable


# Split data into 80/20 train/test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
split_index = int(0.8 * X.shape[0])
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train and evaluate the model
#Neural Network
nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=50, output_size=1, step_size=0.01)
nn.training_nn(X_train, y_train, X_test, y_test, epochs=800)
nn.print_weights()
accuracy_of_nn = nn.accuracy_nn(X_test, y_test)
nn.plot_loss_nn(accuracy_of_nn)

#Decision Tree
dt = DecisionTreeID3(max_depth=10)
dt.fit(X_train, y_train)
accuracy_of_dt = dt.accuracy_of_dt(X_test, y_test)
dt.plot_tree()


