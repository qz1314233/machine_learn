import numpy as np
import h5py
import lr_utils
from PIL import Image
from matplotlib import pyplot as plt

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage


train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = lr_utils.load_dataset()
"""
#测试数据
index = 25 #标签数目
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y_orig[:,index]) + ", it's a '" + classes[np.squeeze(test_set_y_orig[:,index])].decode("utf-8") +  "' picture.")
"""
# 计算训练集、测试集的大小以及图像的大小
m_train = train_set_y_orig.shape[1]
m_test = test_set_y_orig.shape[1]
num_px = train_set_x_orig.shape[1]
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y_orig.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y_orig.shape))

# 转化矩阵
# 整个训练集转为一个矩阵，其中包括num_px*num_py*3行，m_train列
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y_orig.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y_orig.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

# 去中心化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


# 实现反向传播的sigmoid函数
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


# 实现relu函数
def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


# 实现反向传播的relu函数
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01

    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01

    b2 = np.zeros((n_y, 1))

    # 使用断言确保我的数据格式是正确的

    assert (W1.shape == (n_h, n_x))

    assert (b1.shape == (n_h, 1))

    assert (W2.shape == (n_y, n_h))

    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    return parameters


def initialize_parameters_deep(layers_dims):
    np.random.seed(3)

    parameters = {}

    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])

        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        # 确保我要的数据的格式是正确的

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))

        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


def linear_forward(A_prev, W, b):
    """

    实现前向传播的线性部分。

    参数：

        A_prev - 来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）

        W - 权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）

        b - 偏向量，numpy向量，维度为（当前图层节点数量，1）

    返回：

         Z - 激活功能的输入，也称为预激活参数

         cache - 一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递

    """
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    # Please do something

    return Z, linear_cache


def linear_activation_forward(A_prev, W, b, activation):
    """

    实现LINEAR-> ACTIVATION 这一层的前向传播

    参数：

        A_prev - 来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）

        W - 权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）

        b - 偏向量，numpy阵列，维度为（当前层的节点数量，1）

        activation - 选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】

    返回：

        A - 激活函数的输出，也称为激活后的值

        cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递

    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    # Please do something

    return A, cache


def L_model_forward(X, parameters):
    """

    实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION



    参数：

        X - 数据，numpy数组，维度为（输入节点数量，示例数）

        parameters - initialize_parameters_deep（）的输出



    返回：

        AL - 最后的激活值

        caches - 包含以下内容的缓存列表：

                 linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）

                 linear_sigmoid_forward（）的cache（只有一个，索引为L-1）

    """

    # Please do something
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """

    交叉熵误差函数，定义成本函数。

    参数：

        AL - 与标签预测相对应的概率向量，维度为（1，示例数量）

        Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

    返回：

        cost - 交叉熵成本

    """
    m = Y.shape[1]

    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)

    # Please do something

    return cost


def linear_backward(dZ, linear_cache):
    """

    为单层实现反向传播的线性部分（第L层）

    参数：

         dZ - 相对于（当前第l层的）线性输出的成本梯度

         cache - 来自当前层前向传播的值的元组（A_prev，W，b）

    返回：

         dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同

         dW - 相对于W（当前层l）的成本梯度，与W的维度相同

         db - 相对于b（当前层l）的成本梯度，与b维度相同

    """

    # Please do something
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation="relu"):
    """
    实现LINEAR-> ACTIVATION层的后向传播。

    参数：
         dA - 当前层l的激活后的梯度值
         cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
         activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
         db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    """
    # Please do something
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """

    对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播

    参数：

     AL - 概率向量，正向传播的输出（L_model_forward（））

     Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

     caches - 包含以下内容的cache列表：

                 linear_activation_forward（"relu"）的cache，不包含输出层

                 linear_activation_forward（"sigmoid"）的cache



    返回：

     grads - 具有梯度值的字典

              grads [“dA”+ str（l）] = ...

              grads [“dW”+ str（l）] = ...

              grads [“db”+ str（l）] = ...

    """

    # Please do something
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 整除
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if isPlot and i % 100 == 0:
            print("loop %i cost value: %f" % (i, cost))
        if isPlot and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    # 返回parameters
    return parameters


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))
    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("准确度为: " + str(float(np.sum((p == y)) / m)))
    return p


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    np.random.seed(1)
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def print_mislabeled_images(classes, X, y, p):
    """
    绘制预测和实际不同的图像。
        X - 数据集
        y - 实际的标签
        p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[
            y[0, index]].decode("utf-8"))

        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.show()


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
train_set_x_orig.shape
train_set_y.shape
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


parameters = two_layer_model(train_set_x, train_set_y_orig, layers_dims=(n_x, n_h, n_y), num_iterations=2500,
                             print_cost=True, isPlot=True)

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y_orig
test_x = test_x_flatten / 255
test_y = test_set_y_orig

layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, isPlot=True)


def myPredict(img_file_path, label, width, height):
    img = lr_utils.myLoadImgAndResize(img_file_path, width, height)
    plt.imshow(img)

    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.show()
    my_image_x = img.reshape(1, -1).T
    my_image_y = [label]

    predic_res = predict(my_image_x, my_image_y, parameters)
    print(predic_res)
    print("y = " + str(np.squeeze(predic_res)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(predic_res)),].decode("utf-8") + "\" picture")


img_file_path = "datasets/myCat4.jpg"
num_px = 64
myPredict(img_file_path, 1, num_px, num_px)
print("=================训练完成==================")

