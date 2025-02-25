import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from pandas import DataFrame
import numpy as np

SEED = 1145


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):  # sigmoid 计算方式
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):  # sigmoid 导数计算方式
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output, learning_rate):
        error_output = y - output
        nabla_output = error_output * self.sigmoid_derivative(self.output_input)

        error_hidden = np.dot(nabla_output, self.weights_hidden_output.T)
        nabla_hidden = error_hidden * self.sigmoid_derivative(self.hidden_input)

        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, nabla_output)
        self.bias_output += learning_rate * np.sum(nabla_output, axis=0, keepdims=True)

        self.weights_input_hidden += learning_rate * np.dot(X.T, nabla_hidden)
        self.bias_hidden += learning_rate * np.sum(nabla_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (y - output) ** 2)
            self.backward(X, y, output, learning_rate)
            # if epoch % 100 == 0:
                # print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):

        return self.forward(X)


def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels


iris = datasets.load_iris()
df = DataFrame(iris.data, columns=iris.feature_names)
df["target"] = list(iris.target)
X = df.iloc[:, 0:4].values
Y = df.iloc[:, 4].values

mlp_accuracies = []
manual_accuracies = []

for run in range(100):
    current_seed = SEED + run

    random.seed(current_seed)
    np.random.seed(current_seed)

    print("round", run)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED + run, test_size=0.2)

    sc = StandardScaler()
    sc.fit(X)
    standard_train = sc.transform(X_train)
    standard_test = sc.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(20,),
        activation='relu',
        solver='adam',
        max_iter=2000,
        learning_rate_init=0.001,
        alpha=0.001,
        batch_size=20,
        random_state=SEED + run,
    )

    mlp.fit(standard_train, Y_train)
    mlp_result = mlp.predict(standard_test)
    mlp_accuracy = accuracy_score(Y_test, mlp_result)
    mlp_accuracies.append(mlp_accuracy)

    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = len(np.unique(Y_train))
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    Y_train_encoded = one_hot_encode(Y_train)

    nn.train(standard_train, Y_train_encoded, epochs=2000, learning_rate=0.01)

    nn_predictions = nn.predict(standard_test)

    predicted_classes = np.argmax(nn_predictions, axis=1)

    manual_accuracy = accuracy_score(Y_test, predicted_classes)
    manual_accuracies.append(manual_accuracy)

average_mlp_accuracy = np.mean(mlp_accuracies)
average_manual_accuracy = np.mean(manual_accuracies)

print(f"MLPClassifier 平均准确率 over 100 runs: {average_mlp_accuracy:.4f}")
print(f"手动实现的神经网络 平均准确率 over 100 runs: {average_manual_accuracy:.4f}")
