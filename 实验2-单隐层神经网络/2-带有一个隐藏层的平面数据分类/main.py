from simplenn import SimpleNN
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X, Y = load_planar_dataset()
    nn = SimpleNN(2, 4, 1)
    nn.nn_model(X, Y, iterations=10000, learning_rate=0.5)
    # 绘制边界
    plot_decision_boundary(lambda x: nn.predict(x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    predictions = nn.predict(X)
    print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    # plt.figure(figsize=(16, 32))
    #
    # hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
    # for i, n_h in enumerate(hidden_layer_sizes):
    #     nn = SimpleNN(2, n_h, 1)
    #     plt.subplot(5, 2, i + 1)
    #     plt.title('Hidden Layer of size %d' % n_h)
    #     parameters = nn.nn_model(X, Y, iterations=5000, learning_rate=0.05)
    #     plot_decision_boundary(lambda x: nn.predict(x.T), X, Y)
    #     predictions = nn.predict(X)
    #     accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    #     print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
