from tensorflow.compat.v1.keras.datasets.mnist import load_data
import numpy as np
import time
import matplotlib.pyplot as plt


def relu(x):
    return np.where(x < 0, 0, x)


def relu_grad(x):
    return np.where(x < 0, 0.0, 1.0)


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
    # AL = AL + 0.0000000001
    # last_dA = - (Y / AL)
    last_dA = AL
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
            dZ = last_dA * sigmoid_grad(temp_Z)
        elif temp_act == 'relu':
            dZ = last_dA * relu_grad(temp_Z)
        elif temp_act == 'softmax':
            # dZ = last_dA * softmax_grad(temp_Z)
            dZ = last_dA - Y
        else:
            dZ = last_dA
        dW = 1. / m * np.dot(dZ, temp_A.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
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


def dict2vector(params, layers_dims):
    '''
    将所有参数拉直并拼接成一个大的1 * n的向量并返回

    :param params: 包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    :param layers_dims: 每一层输入的维度，第一个数字表示输入层
    :return: 返回由W1,b1 到 Wn, bn 组成的一个大向量
    '''
    theta = None
    for i in range(1, len(layers_dims)):
        w = params['W' + str(i)].copy()
        w = w.reshape(1, -1)
        b = params['b' + str(i)].copy()
        b = b.reshape(1, -1)
        if theta is None:
            theta = np.concatenate((w, b), axis=1)
        else:
            theta = np.concatenate((theta, w), axis=1)
            theta = np.concatenate((theta, b), axis=1)
    return theta


def vector2dict(theta, layers_dims):
    '''
    将拉直后的参数向量，重新整合成参数字典

    :param theta: 拉直后1 * n的参数向量
    :param layers_dims: 每一层输入的维度，第一个数字表示输入层
    :return: 返回按照神经网络结构重新整合的包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    '''
    params = {}
    offset = 0
    for i in range(1, len(layers_dims)):
        w_dims = layers_dims[i] * layers_dims[i - 1]
        w = theta[:, offset:(offset + w_dims)].copy()
        w = w.reshape(layers_dims[i], layers_dims[i - 1])
        offset += w_dims
        params['W' + str(i)] = w
        b_dims = layers_dims[i] * 1
        b = theta[:, offset:(offset + b_dims)].copy()
        b = b.reshape(layers_dims[i], 1)
        offset += b_dims
        params['b' + str(i)] = b
    return params


def grads2vector(grads, layers_dims):
    '''
    将所有的梯度拉直并拼接成一个大的1 * n的向量并返回

    :param grads: 保存每一层权重参数梯度dWn和偏置参数梯度dbn的梯度字典
    :param layers_dims: 每一层输入的维度，第一个数字表示输入层
    :return: 返回由dW1,db1 到 dWn, dbn 组成的一个大向量
    '''
    gradients = None
    for i in range(1, len(layers_dims)):
        dw = grads['dW' + str(i)].copy()
        dw = dw.reshape(1, -1)
        db = grads['db' + str(i)].copy()
        db = db.reshape(1, -1)
        if gradients is None:
            gradients = np.concatenate((dw, db), axis=1)
        else:
            gradients = np.concatenate((gradients, dw), axis=1)
            gradients = np.concatenate((gradients, db), axis=1)
    return gradients


def gradient_check_n(parameters, gradients, layers_dims, X, Y, epsilon=1e-7):
    '''
    进行梯度检测

    :param parameters: 包括每一层权重矩阵Wn和偏置矩阵bn的参数字典
    :param gradients: 保存每一层权重参数梯度dWn和偏置参数梯度dbn的梯度字典
    :param layers_dims: 每一层输入的维度，第一个数字表示输入层
    :param X: 输入
    :param Y: 真实值
    :param epsilon: 允许的最大差异
    :return: 返回近似梯度与实际计算梯度的差异
    '''

    parameters_values = dict2vector(parameters, layers_dims)
    grad = grads2vector(gradients, layers_dims)
    num_parameters = parameters_values.shape[1]
    print('number of parameters: ' + str(num_parameters))
    J_plus = np.zeros((1, num_parameters))
    J_minus = np.zeros((1, num_parameters))
    gradapprox = np.zeros((1, num_parameters))

    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[0, i] = thetaplus[0, i] + epsilon
        yhat, _ = forward_prop(X, vector2dict(thetaplus, layers_dims), activation_functions)
        J_plus[0, i] = compute_cost(yhat, Y)
        # print('J_plus ' + str(i) + ' is: ' + str(J_plus[0, i]))

        thetaminus = np.copy(parameters_values)
        thetaminus[0, i] = thetaminus[0, i] - epsilon
        yhat, _ = forward_prop(X, vector2dict(thetaminus, layers_dims), activation_functions)
        J_minus[0, i] = compute_cost(yhat, Y)
        # print('J_minus ' + str(i) + ' is: ' + str(J_minus[0, i]))

        gradapprox[0, i] = (J_plus[0, i] - J_minus[0, i]) / (2 * epsilon)
        # print('gradapprox ' + str(i) + ' is: ' + str(gradapprox[0, i]))

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference, (grad - gradapprox)


(x_train, y_train), (x_test, y_test) = load_data(r'D:\PythonWorkspace\TFTryout\mnist.npz')
# x_train.shape = (60000, 28, 28)
# y_train.shape = (60000, )
# x_test.shape = (10000, 28, 28)
# y_test.shape = (10000,)

# preprocess: normalize and roll up
x_train_pro = x_train / 255.0
x_test_pro = x_test / 255.0
x_train_pro = np.reshape(x_train_pro, (x_train_pro.shape[0], -1)).T
y_train_lab = np.reshape(y_train, (y_train.shape[0], 1)).T
x_test_pro = np.reshape(x_test_pro, (x_test_pro.shape[0], -1)).T
y_test_lab = np.reshape(y_test, (y_test.shape[0], 1)).T

# 扩展训练集输出
y_train_ext = np.zeros((10, y_train_lab.shape[1]))
for i in range(len(y_train_lab[0])):
    y_train_ext[y_train_lab[0, i], i] = 1.0

# 定义4层神经网络 100 -> 50 -> 20 -> 10
# layers_dims = [x_train_pro.shape[0], 100, 50, 20, 10]
layers_dims = [x_train_pro.shape[0], 128, 10]
parameters = initialize_parameters(layers_dims)

# 定义每一层的激活函数
# activation_functions = ['relu', 'relu', 'relu', 'sigmoid']
activation_functions = ['relu', 'softmax']
# activation_functions = ['relu', 'sigmoid']
# activation_functions = ['sigmoid', 'sigmoid', 'sigmoid', 'softmax']

# train_nums = 10001
learning_rate = 0.1
costs = []
accs = []
batch_size = 64
epochs = 5
for i in range(1, epochs+1):
    print("Now Start Epoch " + str(i))
    # 按照batch_size切分训练数据
    batch_num = x_train_pro.shape[1] // batch_size
    if x_train_pro.shape[1] % batch_size > 0:
        batch_num += 1
    for j in range(batch_num):
        if j == batch_num - 1:
            x_train_batch = x_train_pro[:, (j * batch_size):]
            y_train_batch = y_train_ext[:, (j * batch_size):]
            y_train_lab_batch = y_train_lab[:, (j * batch_size):]
        else:
            x_train_batch = x_train_pro[:, (j*batch_size):((j+1)*batch_size)]
            y_train_batch = y_train_ext[:, (j * batch_size):((j + 1) * batch_size)]
            y_train_lab_batch = y_train_lab[:, (j * batch_size):((j + 1) * batch_size)]
        yhat, caches = forward_prop(x_train_batch, parameters, activation_functions)
        assert yhat.shape == y_train_batch.shape
        grads = back_prop(y_train_batch, caches, parameters, activation_functions)
        assert len(grads) == len(parameters)
        parameters = update_params(parameters, grads, learning_rate)

        if j % 50 == 0:
            cost = compute_cost(yhat, y_train_batch)
            print("After training " + str(j) + " times, cost is " + str(cost))
            yhat_train, _ = forward_prop(x_train_batch, parameters, activation_functions)
            train_acc = cal_accuracy(yhat_train, y_train_lab_batch)
            print("Accuracy on Training set is: " + str(train_acc))
            costs.append(cost)
            accs.append(train_acc)

    print("Epoch " + str(i) + "/" + str(epochs) + " is Done...")
    yhat_train, _ = forward_prop(x_train_pro, parameters, activation_functions)
    train_acc = cal_accuracy(yhat_train, y_train_lab)
    print("Accuracy on Training set is: " + str(train_acc))

    yhat_test, _ = forward_prop(x_test_pro, parameters, activation_functions)
    test_acc = cal_accuracy(yhat_test, y_test_lab)
    print("Accuracy on Test set is: " + str(test_acc))

print("Training is Done")
plt.plot(costs)
plt.title("Costs during training")
plt.show()
plt.plot(accs)
plt.title("Accuracies during training")
plt.show()

yhat_train, _ = forward_prop(x_train_pro, parameters, activation_functions)
train_acc = cal_accuracy(yhat_train, y_train_lab)
print("Accuracy on Training set is: " + str(train_acc))

yhat_test, _ = forward_prop(x_test_pro, parameters, activation_functions)
test_acc = cal_accuracy(yhat_test, y_test_lab)
print("Accuracy on Test set is: " + str(test_acc))


