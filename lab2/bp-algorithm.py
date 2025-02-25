import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from pandas import DataFrame
import numpy as np

SEED = 1145
random.seed(SEED)
np.random.seed(SEED)

iris = datasets.load_iris()
df = DataFrame(iris.data, columns=iris.feature_names)
df["target"] = list(iris.target)
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED, test_size=0.2)

sc = StandardScaler()
sc.fit(X)
standard_train = sc.transform(X_train)
standard_test = sc.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(25,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size=20,
    random_state=SEED,
)

mlp.fit(standard_train, Y_train)

result = mlp.predict(standard_test)


print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(result))
print("预测的准确率为：", mlp.score(standard_test, Y_test))
print("层数为：", mlp.n_layers_)
print("迭代次数为：", mlp.n_iter_)
print("损失为：", mlp.loss_)
print("激活函数为：", mlp.out_activation_)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
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
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels

input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(Y_train))
nn = NeuralNetwork(input_size, hidden_size, output_size)

Y_train_encoded = one_hot_encode(Y_train)

print("training.......")
nn.train(standard_train, Y_train_encoded, epochs=1000, learning_rate=0.01)

predictions = nn.predict(standard_test)

accuracy = accuracy_score(Y_test, np.argmax(predictions, axis=1))

print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(np.argmax(predictions, axis=1)))
print("预测的准确率为：", accuracy)
