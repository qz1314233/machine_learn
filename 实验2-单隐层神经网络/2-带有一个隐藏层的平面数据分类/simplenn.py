import numpy
import numpy as np
import matplotlib.pyplot as plt


class SimpleNN(object):
    def __init__(self, input_layer, hidden_layer, output_layer):
        """
        初始化神经网络，这里的隐藏层只有一层
        :param input_layer: 输入层神经元个数
        :param hidden_layer: 隐藏层神经元个数
        :param output_layer: 输出层神经元个数
        """
        self.w = []
        self.b = []
        self.z = []
        self.a = []
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.initialize_parameters()

    def initialize_parameters(self):
        w1 = np.random.randn(self.hidden_layer, self.input_layer)
        w2 = np.random.randn(self.output_layer, self.hidden_layer)

        assert (w1.shape == (self.hidden_layer, self.input_layer))
        assert (w2.shape == (self.output_layer, self.hidden_layer))

        self.w = [w1, w2]

        b1 = numpy.zeros(shape=(self.hidden_layer, 1))
        b2 = numpy.zeros(shape=(self.output_layer, 1))

        assert (b1.shape == (self.hidden_layer, 1))
        assert (b2.shape == (self.output_layer, 1))

        self.b = [b1, b2]

    def sigmoid(self, z):
        """
        sigmoid激活函数，用于output层
        :param z: 输入
        :return:
        """
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, A, Y):
        """
        计算损失函数
        :param A: 预测结果
        :param Y: 实际结果
        :return:
        """
        m = Y.shape[1]
        total_loss = (-1) * np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))
        cost = (1 / m) * np.sum(total_loss)
        cost = float(np.squeeze(cost))
        assert (isinstance(cost, float))

        return cost

    def forward(self, X):
        """
        前向传播
        :param X: 样本集
        :return:

        """
        m = X.shape[1]
        Z1 = np.dot(self.w[0], X) + self.b[0]
        A1 = np.tanh(Z1)

        Z2 = np.dot(self.w[1], A1) + self.b[1]
        A2 = self.sigmoid(Z2)

        self.z = [Z1, Z2]
        self.a = [X, A1, A2]

    def backward(self, Y, learning_rate):
        """
        反向传播
        :param Y: 实际标签值
        :param learning_rate: 学习率
        :return:
        """
        sample_num = Y.shape[1]
        dZ2 = self.a[2] - Y
        dW2 = (1 / sample_num) * np.dot(dZ2, self.a[1].T)
        db2 = (1 / sample_num) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(self.w[1].T, dZ2), 1 - np.power(self.a[1], 2))
        dW1 = (1 / sample_num) * np.dot(dZ1, self.a[0].T)
        db1 = (1 / sample_num) * np.sum(dZ1, axis=1, keepdims=True)

        self.w[0] = self.w[0] - learning_rate * dW1
        self.w[1] = self.w[1] - learning_rate * dW2

        self.b[0] = self.b[0] - learning_rate * db1
        self.b[1] = self.b[1] - learning_rate * db2

    def predict(self, X):
        """
        预测函数
        :param X: 输入数据
        :return:
        """
        self.forward(X)
        return np.round(self.a[2])

    def nn_model(self, X, Y, learning_rate, iterations):
        """
        主控函数
        :param X: 输入样本
        :param Y: 实际标签值
        :param learning_rate: 学习率
        :param iterations: 迭代次数
        :return:
        """
        for i in range(0, iterations):
            self.forward(X)
            self.backward(Y, learning_rate)
            cost = self.compute_loss(self.a[2], Y)
            if i % 1000 == 0:
                print("第 ", i, " 次循环，成本为：" + str(cost))

