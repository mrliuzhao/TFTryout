from tensorflow.compat.v1.keras.datasets.mnist import load_data
import numpy as np


def relu(x):
    return np.where(x < 0, x, x)


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


# ---------------- Copy from Coursera ----------------

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambd / m) * W3
    ### END CODE HERE ###
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    ### END CODE HERE ###
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd / m) * W1
    ### END CODE HERE ###
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """

    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above.
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = (D1 < keep_prob).astype(int)  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = (D2 < keep_prob).astype(int)  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2  # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.

    Arguments:
    X -- training set for m examples
    Y -- labels for m examples
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)

    Returns:
    cost -- the cost function (logistic cost for one example)
    """

    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()

    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


# a function "dictionary_to_vector()" for you. It converts the "parameters" dictionary into a vector called "values", obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them
#
# The inverse function is "vector_to_dictionary" which outputs back the "parameters" dictionary.
#
# We have also converted the "gradients" dictionary into a vector "grad" using gradients_to_vector(). You don't need to worry about that.

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))  # Step 3
        ### END CODE HERE ###

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))  # Step 3
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

