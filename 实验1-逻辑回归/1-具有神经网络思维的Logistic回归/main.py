import numpy as np  # numpy 是使用Python进行科学计算的基础包。
import matplotlib.pyplot as plt  # 是Python中著名的绘图库。
import h5py
from numpy.core.fromnumeric import shape  # 基于NumPy来做高等数学、信号处理、优化、统计和许多其它科学任务的拓展库。
import scipy  ##ython提供读取HDF5二进制数据格式文件的接口，本次的训练及测试图片集是以HDF5储存的。
from PIL import Image  # (Python Image Library) 为 Python提供图像处理功能
from scipy import ndimage
from lr_utils import load_dataset  # 用来导入数据集的

# %matplotlib inline   #设置matplotlib在行内显示图片

# 温馨提示：如果该作业在本地运行，该数据集的代码保存在lr_utils.py文件，并和当前项目保存在一个文件夹下
"""
    train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
    train_set_y_orig ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
    test_set_y_orig ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
"""
# 导入数据
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
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


# sigmoid()函数实现
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


"""
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(9.2) = " + str(sigmoid(9.2)))
"""


# 初始化参数（w,  b）
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)参数的数量

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros(shape=(dim, 1))  # 初始化 w 为 (dim行，1列) 的向量
    b = 0
    assert (w, shape == (dim, 1))  # 判断 w 的shape是否为 (dim, 1), 不是则终止程序
    assert (isinstance(b, float) or isinstance(b, int))  # 判断 b 是否是float或者int类型

    return w, b


"""
w,b = initialize_with_zeros(5)
print("w=", w)
print("b=", b)
"""


# 前向传播和后向传播
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation
    """
    m = X.shape[1]  # 样例个数

    # 前向传播(Forward Propagation)
    A = sigmoid(np.dot(w.T, X) + b)  # 计算 activation , A 的 维度是 (m, m)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # 计算 cost; Y == yhat(1, m)
    # 反向传播(Backward Propagation)
    dw = (1 / m) * np.dot(X, (A - Y).T)  # 计算 w 的导数
    db = (1 / m) * np.sum(A - Y)  # 计算 b 的导数

    assert (dw.shape == w.shape)  # 减少bug出现
    assert (db.dtype == float)  # db 是一个值
    cost = np.squeeze(cost)  # 压缩维度,(从数组的形状中删除单维条目，即把shape中为1的维度去掉)，保证cost是值
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


"""
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y) 
print ("dw = " + str(grads["dw"]))  
print ("db = " + str(grads["db"]))  
print ("cost = " + str(cost))
"""


# Optimization(最优化)
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop  优化循环的迭代次数
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- 包含权重w和偏差b的字典
    grads -- 字典包含权重的梯度和相对于代价函数的偏差
    costs -- 在优化过程中计算的所有成本的列表，这将用于绘制学习曲线.

    小贴士:
    你基本上需要写下两个步骤并迭代它们:
        1)计算当前参数的代价和梯度。使用传播()。
        2)对w和b使用梯度下降规则更新参数。
    """
    iterations = []
    costs = []
    for i in range(num_iterations):

        # Cost and gradient calculation(成本和梯度计算)
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads(获取导数)
        dw = grads["dw"]
        db = grads["db"]

        # update rule (更新 参数)
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db

        # Record the costs (每一百次记录一次 cost)
        if i % 100 == 0:
            iterations.append(i)
            costs.append(cost)
        # Print the cost every 100 training examples (如果需要打印则每一百次打印一次)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    # 记录 迭代好的参数 (w, b)
    params = {"w": w,
              "b": b}

    # 记录当前导数(dw, db), 以便下次继续迭代
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs, iterations


"""
params, grads, costs = optimize(w, b, X, Y, num_iterations= 200, learning_rate = 0.009, print_cost = True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
"""


# 预测函数 predict
# 1.	计算 Y_hat = A = sigmod(w.T X + b)
# 2.	转换 a 为 0 (如果 activation <= 0.5) 或者 1 (如果activation > 0.5)，存储预测值在 向量Y_prediction中
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计算向量A，预测图片中出现一只猫的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a[0,i]转换为实际预测p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# print("predictions = " + str(predict(w, b, X)))

# 合并所有函数在一个model()里
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- 训练集training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- 训练标签training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.包含关于模型的信息的字典
    """
    # initialize parameters with zeros (初始化参数(w, b))
    w, b = initialize_with_zeros(X_train.shape[0])  # num_px*num_px*3

    # Gradient descent (前向传播和后向传播 同时 梯度下降更新参数)
    parameters, grads, costs, iterations = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    print("iterations", iterations)
    # Retrieve parameters w and b from dictionary "parameters"(获取参数w, b)
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (使用测试集和训练集进行预测)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors (训练/测试误差: (100 - mean(abs(Y_hat - Y))*100 )
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "iterations": iterations,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations=2000, learning_rate=0.005,
          print_cost=True)

# 图片分类错误的例子
index = 5
# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# test_set_y[0, index]：测试集里标签;  classes[int(d["Y_Prediction_test"][0, index])]：预测值
print("y = " + str(test_set_y_orig[0, index]) + ", you predicted that it is a \"" + classes[
    int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")
# Plot learning curve (with costs) 绘制成本图
iterations = np.array(np.squeeze(d['iterations']))
print("iterations===", iterations)
costs = np.array(np.squeeze(d['costs']))
print("costs===", costs)
plt.plot(iterations, costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# 改变学习率调试
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations=1500,
                           learning_rate=i, print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()



print("====================测试model====================")
#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()