import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Part 1: Single-Layer Neural Network with Gradient Descent
class SingleLayerNN:
    def __init__(self, input_size, step_size=0.000005):
        self.step_size = step_size
        self.train_loss = []
        self.test_loss = []

    def sigmoid(self, z):      #activation function
        z = np.clip(z, -700, 700)
        return 1 / (1 + np.exp(-z))

    def gradient_sigmoid(self, z):      #derivative of the activation function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cross_entropy_loss(self, y_true, y_pred):      #loss function
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def training(self, X_train, y_train, X_test, y_test, epochs=500):       #training the model with gradient descent
        X_train_with_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        X_test_with_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
        w_0 = np.random.rand(X_train_with_bias.shape[1], len(X_test_with_bias[1])) * 0.1
        w_1 = np.random.rand(len(X_test_with_bias[1]), 1) * 0.1

        for epoch in range(epochs):     #loop through the epochs
            w_0, w_1 = self.gradient_descent(X_train_with_bias, y_train, w_0, w_1, self.step_size)
            train_pred = self.predict(X_train, w_0, w_1)
            test_pred = self.predict(X_test, w_0, w_1)
            train_loss = np.mean(self.cross_entropy_loss(y_train, train_pred))
            test_loss = np.mean(self.cross_entropy_loss(y_test, test_pred))
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)


        self.weights_0 = w_0
        self.weights_1 = w_1


    def gradient_descent(self, a_0, y, w_0, w_1, eta):      #gradient descent function
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

    def predict(self, X, w_0_trained, w_1_trained):     #predict function
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        a_1 = np.dot(X_with_bias, w_0_trained)
        z_1 = self.sigmoid(a_1)
        a_2 = np.dot(z_1, w_1_trained)
        y_predicted = self.sigmoid(a_2)
        return y_predicted

    def accuracy(self, X, y, w_0_trained, w_1_trained):     #accuracy function
        y_predicted = self.predict(X, w_0_trained, w_1_trained)
        y_predicted = np.round(y_predicted)
        return np.sum(y_predicted == y) / len(y)

    def plot_loss(self):     #plotting the loss curve for train and test data
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


# Part 2: decision tree with ID3 algorithm with entropy potential function

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        counts = np.sum(y == 1) #######
        q = counts / len(y)
        return -0.5*(q * np.log2(q) + (1-q)*np.log2(1-q)) if q > 0 and q < 1 else 0

    def information_gain(self, X_column, y, threshold):
        before_split_entropy = self.entropy(y)  # Calculate the entropy before the split
        left_indices = X_column <= threshold  # Indices for the left side after the split
        right_indices = X_column > threshold  # Indices for the right side after the split
        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return 0  # If either side is empty, return 0 gain
        n = len(y)
        n_left = sum(left_indices)
        n_right = sum(right_indices)
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
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
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

    def predict_sample(self, sample, tree):
        if not isinstance(tree, tuple):
            return 'Not a decision tree'
        predicator, threshold, left_subtree, right_subtree = tree
        if sample[predicator] <= threshold:
            return self.predict_sample(sample, left_subtree)
        else:
            return self.predict_sample(sample, right_subtree)

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])


    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.sum(predictions == y) / len(y)

    def plot_tree(self):
        def add_edges(tree, graph, parent=None, edge_label=""):
            if not isinstance(tree, tuple):
                graph.add_node(str(tree))
                if parent:
                    graph.add_edge(parent, str(tree), label=edge_label)
                return
            col, threshold, left_subtree, right_subtree = tree
            node_label = f"X[{col}] <= {threshold}"
            graph.add_node(node_label)
            if parent:
                graph.add_edge(parent, node_label, label=edge_label)
            add_edges(left_subtree, graph, node_label, "True")
            add_edges(right_subtree, graph, node_label, "False")

        graph = nx.DiGraph()
        add_edges(self.tree, graph)
        pos = nx.spring_layout(graph)
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
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
#nn = SingleLayerNN(input_size=X_train.shape[1])
#nn.training(X_train, y_train, X_test, y_test)

# Train and evaluate the model

#accuracy = nn.accuracy(X_test, y_test, nn.weights_0, nn.weights_1)
#print(f'Accuracy: {accuracy}')
#nn.plot_loss()


dt = DecisionTreeID3(max_depth=10)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
accuracy = dt.accuracy(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot the tree
dt.plot_tree()