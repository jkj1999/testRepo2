import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
n_train = 50
# torch.rand(n)生成0-1之间的n个随机数
# torch.sort()的返回值为一个Union,我们只关心排好序的x_train,不关心其余的变量如排序后的索引等其他信息,使用_占位符变量
x_train, _ = torch.sort(torch.rand(n_train) * 5)


# 真实生成函数
def f(x):
    return 2 * torch.sin(x) + x ** 0.8


# torch.normal(mean,std,Union[size,...])指定size时,参数要求是元组类型且第一个变量为size大小,通过(n,)定义只关心size大小的一个元组
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出,带有高斯噪声
x_test = torch.arange(0, 5, 0.1)  # 50个测试样本
n_test = len(x_test)  # 测试样本数
y_truth = f(x_test)  # 测试样本的真实输出


def plot_kernel_reg(y_hat):
    # 绘制测试数据的实际值和预测值
    # legend指定图例标签
    # xlim, ylim设置横纵轴的坐标轴范围
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])

    # 绘制训练数据
    # alpha指定数据点的透明度,0-1之间,0表示完全透明
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)


# 平均汇聚
# 创建一个与测试集相同大小的预测值张量，其中的每个元素都是训练集目标值的平均值
# torch.repeat_interleave() 重复插入tensor的元素
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
d2l.plt.show()

# 非参数注意力汇聚
# X_repeat的形状:(n_test,n_train)
# e.g.: 3 * 3
# tensor([[0, 0, 0],
#         [1, 1, 1],
#         [2, 2, 2]])
# 每一行都包含着相同的测试输入（例如：同样的查询）
# 将测试集 x_test 中的每个元素都重复插入 n_train 次,并重设张量形状,-1 自动计算维度大小,每个样本应该包含 n_train 个特征
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train)
# 每一行代表在给定当前行对应的x_test一个输入值时,每个y_train对应于该输入值的输出的注意力权重
# 将权重乘以每个y_train相加即得到该行输入对应的输出
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
# torch.matmul()执行张量之间的矩阵乘法
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
d2l.plt.show()
# 从绘制的结果会发现新的模型预测线是平滑的,比平均汇聚的预测更接近真实,因为NW核回归具有一致性的优点,当数据量足够,模型就会收敛到最优
# 在上述非参数注意力权重的例子中,测试数据的输入相当于query,训练数据的输入相当于key

# weights heatmap
# d2l.show_heatmaps()用于可视化热图,通常用来显示数据矩阵中的值,颜色越深表示值越大
# unsqueeze(0)用于插入新的维度
# 假设原矩阵为(n,m)
# unsqueeze(0) → (1,n,m) 单个样本或批处理中的一个样本
# unsqueeze(0).unsqueeze(0) → (1,1,n,m)  单个样本或批处理中的一个通道
# 一般深度学习中处理的tensor张量都是4个维度,所以需要以这种方式做reshape
# 也可以替换为attention_weights.reshape((1, 1, len(x_test),len(x_train)))
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
d2l.plt.show()

# 带参数注意力汇聚

# 小批量矩阵乘法
# 第一个小批量数据包含n个矩阵,形状a*b, 第二个小批量数据包含n个矩阵,形状b*c, 将n对矩阵分别两两相乘,得到n个a*c矩阵
# torch.bmm(X,Y) batchMatricesMultiply
# X = torch.ones((2, 1, 4))
# Y = torch.ones((2, 4, 6))
# print(torch.bmm(X, Y).shape)
# 在注意力机制中,使用小批量矩阵乘法计算小批量数据中的加权平均值

# 定义NW核回归的带参数版本
class NWKernelRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 将张量标记为模型的可训练参数 (1,)即创建一个包含一个元素的一维张量
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # 扩展queries,使其具有与keys相同的形状
        # queries和attention_weights的形状为(查询个数,“键－值”对个数)
        # 先将queries中的每个query重复keys.shape[1]次(实际为键值对的个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数,“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# train
# 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键-值”对进行计算，从而得到其对应的预测输出

# X_tile的形状:(n_train,n_train)，行与行之间相同，包括所有训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，行与行之间相同，包括所有训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train','n_train'-1) 每n行缺少X_tile矩阵的第n个元素
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train','n_train'-1) 每n行缺少Y_tile矩阵的第n个元素
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step() # 使用优化器执行一步参数更新
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
d2l.plt.show() # show animator

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
# tensor.unsqueeze() 将张量维度扩展
# tensor.detach() 使张量不再具有梯度信息，减少内存消耗，加快计算
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
d2l.plt.show()

# 注:模型是使用train_set进行回归训练，然后使用训练好的模型拟合test_set