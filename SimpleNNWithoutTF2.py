from tensorflow.compat.v1.keras.datasets.mnist import load_data
import numpy as np


def relu(x):
    return np.where(x < 0, 0.01*x, x)


def relu_grad(x):
    return np.where(x < 0, 0.01, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    xsig = sigmoid(x)
    return xsig * (1 - xsig)


def softmax(x):
    x_max = np.amax(x, axis=0, keepdims=True)
    xtemp = x - x_max
    x_exp = np.exp(xtemp)
    exp_sum = np.sum(x_exp, axis=0, keepdims=True)
    return x_exp / exp_sum


def softmax_grad(x):
    xsoft = softmax(x)
    return xsoft * (1 - xsoft)


def grad_fit(x, func):
    epsilon = 0.0001
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


def initialize_parameters(layer_dims):
    '''
    初始化全连接神经网络参数

    :param layer_dims: 每一层输入的维度，第一个数字表示输入层
    :return: params，一个字典，params['Wn']表示第n层权重矩阵，params['bn']表示第n层的偏置矩阵
    '''
    params = {}
    for i in range(1, len(layer_dims)):
        # 尝试Xavier Initialization优化
        factor = np.sqrt(1 / layer_dims[i - 1])
        params['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * factor
        params['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return params


def forward_prop(X, params, activate_funcs):
    '''
    Forward Propagation，计算输出层的输出

    :param X: 输入矩阵，每一列代表一个样本
    :param params: 包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    :param activate_funcs: 字符数组，表示每一层使用的激活函数，目前有'sigmoid'和'relu'两种
    :return: yhat，输出层的输出值，10*m；caches，保存每一层的线性输出Zn和激活输出An的缓存字典
    '''

    assert len(params) == 2 * len(activate_funcs)

    caches = {}
    caches['A0'] = X
    last_a = X
    for i in range(1, len(activate_funcs) + 1):
        temp_W = params['W' + str(i)]
        temp_b = params['b' + str(i)]
        temp_act = activate_funcs[i - 1]
        z = np.dot(temp_W, last_a) + temp_b
        caches['Z' + str(i)] = z
        if temp_act == 'sigmoid':
            last_a = sigmoid(z)
        elif temp_act == 'relu':
            last_a = relu(z)
        elif temp_act == 'softmax':
            last_a = softmax(z)
        else:
            last_a = z
        caches['A' + str(i)] = last_a
    return last_a, caches


def compute_cost(yhat, y):
    '''
    计算交叉熵作为损失

    :param yhat: 预测值10*m
    :param y: 真实值的扩展10*m
    :return: 预测值与真实值之间的损失（交叉熵的平均值）
    '''

    assert yhat.shape == y.shape

    m = y.shape[1]
    cost = np.sum(-np.log(yhat + 0.0000000001) * y) / m
    return cost


def back_prop(Y, caches, params, activate_funcs):
    '''
    反向传播，计算出每一层的梯度

    :param Y: 真实值 10 * m
    :param caches: 保存每一层的线性输出Zn和激活输出An的缓存字典
    :param params: 包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    :param activate_funcs: 字符数组，表示每一层使用的激活函数，目前有'sigmoid'和'relu'两种
    :return: grads，保存每一层权重参数梯度dWn和偏置参数梯度dbn的梯度字典
    '''

    L = len(activate_funcs)
    m = Y.shape[1]
    AL = caches['A' + str(L)]
    AL = AL + 0.0000000001
    last_dA = - (Y / AL)
    # epsilon = 0.00001
    # yadd = -Y * np.log(AL + epsilon)
    # ymin = -Y * np.log(AL - epsilon)
    # last_dA = (yadd - ymin) / (2 * epsilon)

    grads = {}
    for i in reversed(range(L)):
        temp_act = activate_funcs[i]
        temp_Z = caches['Z' + str(i + 1)]
        temp_A = caches['A' + str(i)]
        temp_W = params['W' + str(i + 1)]
        if temp_act == 'sigmoid':
            # dZ = last_dA * sigmoid_grad(temp_Z)
            dZ = last_dA * grad_fit(temp_Z, sigmoid)
        elif temp_act == 'relu':
            # dZ = last_dA * relu_grad(temp_Z)
            dZ = last_dA * grad_fit(temp_Z, relu)
        elif temp_act == 'softmax':
            # dZ = last_dA * softmax_grad(temp_Z)
            # dZ = AL - Y
            dZ = last_dA * grad_fit(temp_Z, softmax)
        else:
            dZ = last_dA
        dW = np.dot(dZ, temp_A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        last_dA = np.dot(temp_W.T, dZ)
        grads['dW' + str(i + 1)] = dW
        grads['db' + str(i + 1)] = db
    return grads


def update_params(params, grads, learning_rate = 0.1):
    '''
    根据梯度更新参数值

    :param params: 包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    :param grads: 保存每一层权重参数梯度dWn和偏置参数梯度dbn的梯度字典
    :param learning_rate: 学习率
    :return:返回更新后的参数字典params
    '''
    L = len(params) // 2
    for i in range(1, L + 1):
        new_W = params['W' + str(i)] - learning_rate * grads['dW' + str(i)]
        new_b = params['b' + str(i)] - learning_rate * grads['db' + str(i)]
        params['W' + str(i)] = new_W
        params['b' + str(i)] = new_b
    return params


def cal_accuracy(lastA, y):
    # 取10个神经元中最大的为预测标签
    lastA = np.argmax(lastA, axis=0)
    lastA = np.reshape(lastA, (1, -1))
    return np.count_nonzero(lastA == y) / y.shape[1]


def grad_check(X, Y, params, grads, activate_funcs, epsilon = 0.001):
    '''
    对计算出的梯度进行检查

    :param X: 输入 n * m
    :param Y: 真实标签值 10 * m
    :param params: 包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    :param grads: 保存每一层权重参数梯度dWn和偏置参数梯度dbn的梯度字典
    :param activate_funcs: 字符数组，表示每一层使用的激活函数
    :param epsilon: 模拟计算梯度时增加的值
    :return: 返回包含每一层梯度差异大小（包含dWn和dbn）的字典grad_dist
    '''
    grad_dist = {}
    L = len(activate_funcs)
    for i in range(1, L + 1):
        w_ori = params['W' + str(i)]
        params['W' + str(i)] = w_ori + epsilon
        yhat_add, _ = forward_prop(X, params, activation_functions)
        cost_add = compute_cost(yhat_add, Y)
        params['W' + str(i)] = w_ori - epsilon
        yhat_min, _ = forward_prop(X, params, activation_functions)
        cost_min = compute_cost(yhat_min, Y)
        dwd = (cost_add - cost_min) / (2 * epsilon)
        np.sum(grads['dW' + str(i)])
        grad_dist['W' + str(i)] = (cost_add - cost_min) / (2 * epsilon)
        params['W' + str(i)] = w_ori

        b_ori = params['b' + str(i)]
        params['b' + str(i)] = b_ori + epsilon
        yhat_add, _ = forward_prop(X, params, activation_functions)
        cost_add = compute_cost(yhat_add, Y)
        params['b' + str(i)] = b_ori - epsilon
        yhat_min, _ = forward_prop(X, params, activation_functions)
        cost_min = compute_cost(yhat_min, Y)
        grad_dist['b' + str(i)] = (cost_add - cost_min) / (2 * epsilon)
        params['b' + str(i)] = b_ori

    return grad_dist



(x_train, y_train), (x_test, y_test) = load_data(r'D:\PythonWorkspace\TFTryout\mnist.npz')
# x_train.shape = (60000, 28, 28)
# y_train.shape = (60000, )
# x_test.shape = (10000, 28, 28)
# y_test.shape = (10000,)

# preprocess: normalize and roll up
x_train_pro = x_train / 255.0
x_test_pro = x_test / 255.0
x_train_pro = np.resize(x_train_pro, (x_train_pro.shape[0], -1)).T
y_train_lab = np.resize(y_train, (y_train.shape[0], 1)).T
x_test_pro = np.resize(x_test_pro, (x_test_pro.shape[0], -1)).T
y_test_lab = np.resize(y_test, (y_test.shape[0], 1)).T

# 扩展训练集输出
y_train_ext = np.zeros((10, y_train_lab.shape[1]))
for i in range(len(y_train_lab[0])):
    y_train_ext[y_train_lab[0, i], i] = 1.0

# 定义4层神经网络 100 -> 50 -> 20 -> 10
layers_dims = [x_train_pro.shape[0], 100, 50, 20, 10]
parameters = initialize_parameters(layers_dims)

# 定义每一层的激活函数
# activation_functions = ['relu', 'relu', 'relu', 'sigmoid']
activation_functions = ['relu', 'relu', 'relu', 'softmax']
# activation_functions = ['sigmoid', 'sigmoid', 'sigmoid', 'softmax']

train_nums = 301

for i in range(train_nums):
    yhat, caches = forward_prop(x_train_pro, parameters, activation_functions)
    assert yhat.shape == y_train_ext.shape

    if i % 20 == 0:
        cost = compute_cost(yhat, y_train_ext)
        print("After training " + str(i) + " times, cost is " + str(cost))
        yhat_train, _ = forward_prop(x_train_pro, parameters, activation_functions)
        train_acc = cal_accuracy(yhat_train, y_train_lab)
        print("Accuracy on Training set is: " + str(train_acc))

    grads = back_prop(y_train_ext, caches, parameters, activation_functions)
    assert len(grads) == len(parameters)

    # 每隔一段时间进行一次grad check
    # if i % 20 == 0:

    parameters = update_params(parameters, grads, 0.05)


print("Training is Done")
yhat_train, _ = forward_prop(x_train_pro, parameters, activation_functions)
train_acc = cal_accuracy(yhat_train, y_train_lab)
print("Accuracy on Training set is: " + str(train_acc))

yhat_test, _ = forward_prop(x_test_pro, parameters, activation_functions)
test_acc = cal_accuracy(yhat_test, y_test_lab)
print("Accuracy on Test set is: " + str(test_acc))


