import numpy as np

from tqdm import tqdm

# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy')  # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy')  # (60000, )
y_train = np.eye(10)[y_train]  # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy')  # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy')  # (10000,)
y_val = np.eye(10)[y_val]  # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy')  # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy')  # (10000,)
y_test = np.eye(10)[y_test]  # (10000, 10), one-hot编码


# 定义激活函数
def acti_func(x):
    '''
    relu函数
    '''
    return np.where(x > 0, x, 0)
    #sigmoid
    #return 1. / (1. + np.exp(-x))
    #tanh
    #return np.tanh(x)




def acti_func_prime(x):
    '''
    relu函数的导数
    '''
    return np.where(x > 0, 1, 0)
    #sigmoid_prime
    #return acti_func(x) * (1. - acti_func(x))
    #tanh_prime
    #return 1 - np.square(acti_func(x))





# 输出层激活函数
def f(x):

    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def f_prime(x):
    '''
    softmax
    '''
    s = f(x).reshape(-1,1)
    return np.diagflat(s) - np.dot(s,s.T)
    #理论正确，但实际使用时未采用该方法



# 定义损失函数:交叉熵
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    #交叉熵：
    epsilon = 1e-10
    y_pred = np.clip(y_pred,epsilon,1-epsilon)
    loss = np.mean(-np.sum(y_true*np.log(y_pred),axis=1))
    #一型：(abs)
    #loss = np.mean(np.sum(np.abs(y_pred - y_true), axis=1))
    #二型：(mse)
    #loss = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
    return loss


def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    batch_size = y_true.shape[0]
    #交叉熵：
    return y_pred - y_true
    #计算时已经简化
    #L1:abs
    #return np.sign(y_pred - y_true)/batch_size
    #L2:mse
    #return 2 * (y_pred - y_true)/batch_size


# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / shape[0]), size=shape)


# 定义网络结构:基础三层
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.w1 = init_weights((input_size, hidden_size))
        self.b1 = init_weights((hidden_size,))
        self.w2 = init_weights((hidden_size, output_size))
        self.b2 = init_weights((output_size,))


    def forward(self, x):
        '''
        前向传播
        '''
        z1 = x.dot(self.w1) + self.b1
        a1 = acti_func(z1)
        z2 = a1.dot(self.w2) + self.b2
        y_pred = f(z2)
        return y_pred

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''

        # 前向传播
        z1 = x_batch.dot(self.w1) + self.b1
        a1 = acti_func(z1)
        z2 = a1.dot(self.w2) + self.b2
        y_pred = f(z2)

        # 计算损失和准确率
        loss = loss_fn(y_batch, y_pred)
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        accuracy = np.mean(pred_labels == true_labels)

        # 反向传播
        #首先计算梯度
        y_grad = loss_fn_prime(y_batch,y_pred)#输出层
        #L1、L2
        #dL_dz2 = y_pred * (y_grad - np.sum(y_grad * y_pred, axis=1, keepdims=True))
        #softmax修正
        dL_dz2 = y_grad
        #交叉熵
        z1_grad = dL_dz2.dot(self.w2.T)
        z1_grad *= acti_func_prime(z1)#隐藏层
        #接着更新delta
        dw2 = a1.T.dot(dL_dz2)
        db2 = np.sum(dL_dz2, axis=0)
        dw1 = x_batch.T.dot(z1_grad)
        db1 = np.sum(z1_grad, axis=0)


        # 更新权重
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

        return loss, accuracy


    def evaluate(self,x_val,y_val):
        #验证集进行评估
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(x_val), 64))
        for i in p_bar:
            x_batch = x_val[i:i + 64]
            y_batch = y_val[i:i + 64]
            # 前向传播
            z1 = x_batch.dot(self.w1) + self.b1
            a1 = acti_func(z1)
            z2 = a1.dot(self.w2) + self.b2
            y_pred = f(z2)
            loss = loss_fn(y_batch, y_pred)
            pred_labels = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            accuracy = np.mean(pred_labels == true_labels)
            losses.append(loss)
            accuracies.append(accuracy)
            p_bar.set_description(f'Val: Loss {np.mean(losses):.4f}, Acc {np.mean(accuracies):.4f}')
        return np.mean(losses), np.mean(accuracies)


#进阶：四层
# class Network(object):
#     def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, lr=0.01):
#         # 增加第二层参数
#         self.w1 = init_weights((input_size, hidden_size_1))  # (784, 256)
#         self.b1 = init_weights((hidden_size_1,))
#         self.w2 = init_weights((hidden_size_1, hidden_size_2))  # 新增层 (256, 256)
#         self.b2 = init_weights((hidden_size_2,))
#         self.w3 = init_weights((hidden_size_2, output_size))  # (256, 10)
#         self.b3 = init_weights((output_size,))
#         self.lr = lr
#
#     def forward(self, x):
#         # 修改前向传播流程
#         z1 = x.dot(self.w1) + self.b1
#         a1 = acti_func(z1)
#         z2 = a1.dot(self.w2) + self.b2  # 新增层前向计算
#         a2 = acti_func(z2)  # 新增层激活
#         z3 = a2.dot(self.w3) + self.b3
#         return f(z3)
#
#     def step(self, x_batch, y_batch):
#         # 前向传播（增加中间层记录）
#         z1 = x_batch.dot(self.w1) + self.b1
#         a1 = acti_func(z1)
#         z2 = a1.dot(self.w2) + self.b2  # 新增层前向
#         a2 = acti_func(z2)  # 新增层激活
#         z3 = a2.dot(self.w3) + self.b3
#         y_pred = f(z3)
#
#         #计算损失和准确率
#         loss = loss_fn(y_batch, y_pred)
#         pred_labels = np.argmax(y_pred, axis=1)
#         true_labels = np.argmax(y_batch, axis=1)
#         accuracy = np.mean(pred_labels == true_labels)
#
#         # 反向传播流程修改
#         y_grad = loss_fn_prime(y_batch, y_pred)
#
#         # 第三层梯度（输出层→新增层）
#         dL_dz3 = y_grad
#         dw3 = a2.T.dot(dL_dz3)
#         db3 = np.sum(dL_dz3, axis=0)
#
#         # 第二层梯度（新增层→原隐藏层）
#         z2_grad = dL_dz3.dot(self.w3.T) * acti_func_prime(z2)
#         dw2 = a1.T.dot(z2_grad)
#         db2 = np.sum(z2_grad, axis=0)
#
#         # 第一层梯度
#         z1_grad = z2_grad.dot(self.w2.T) * acti_func_prime(z1)
#         dw1 = x_batch.T.dot(z1_grad)
#         db1 = np.sum(z1_grad, axis=0)
#
#         # 更新所有参数
#         self.w3 -= self.lr * dw3
#         self.b3 -= self.lr * db3
#         self.w2 -= self.lr * dw2
#         self.b2 -= self.lr * db2
#         self.w1 -= self.lr * dw1
#         self.b1 -= self.lr * db1
#
#         return loss, accuracy
#
#     def evaluate(self, x_val, y_val):
#         losses = []
#         accuracies = []
#         p_bar = tqdm(range(0, len(x_val), 64))
#         for i in p_bar:
#             x_batch = x_val[i:i + 64]
#             y_batch = y_val[i:i + 64]
#             # 修改前向传播（添加新增层）
#             z1 = x_batch.dot(self.w1) + self.b1
#             a1 = acti_func(z1)
#             z2 = a1.dot(self.w2) + self.b2
#             a2 = acti_func(z2)
#             z3 = a2.dot(self.w3) + self.b3
#             y_pred = f(z3)
#             loss = loss_fn(y_batch, y_pred)
#             pred_labels = np.argmax(y_pred, axis=1)
#             true_labels = np.argmax(y_batch, axis=1)
#             accuracy = np.mean(pred_labels == true_labels)
#             losses.append(loss)
#             accuracies.append(accuracy)
#             p_bar.set_description(f'Val: Loss {np.mean(losses):.4f}, Acc {np.mean(accuracies):.4f}')
#
#         return np.mean(losses), np.mean(accuracies)



if __name__ == '__main__':
    # 训练网络
    #三层基础
    net = Network(input_size=784, hidden_size=128, output_size=10, lr=0.01)
    #四层进阶
    #net = Network(input_size=784, hidden_size_1=256, hidden_size_2=36, output_size=10, lr=0.01)
    for epoch in range(10):
        train_losses = []
        train_accs = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i + 64]
            y_batch = y_train[i:i + 64]
            loss, acc = net.step(x_batch, y_batch)
            train_losses.append(loss)
            train_accs.append(acc)
            p_bar.set_description(
                f'Epoch {epoch + 1}, Train: Loss {np.mean(train_losses):.4f}, Acc {np.mean(train_accs):.4f}')

        #可视化
        val_loss, val_acc = net.evaluate(X_val, y_val)
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Train Loss: {np.mean(train_losses):.4f}, Train Acc: {np.mean(train_accs):.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    test_loss,test_acc = net.evaluate(X_test,y_test)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')