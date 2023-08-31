import torch
from torch import nn
from d2l import torch as d2l

# 生成数据
# 使用正弦函数和一些可加性噪声生成序列数据,时间步为1,2,...,1000
T = 1000  # 总共产生1000个点
# time是从1到1000
time = torch.arange(1, T + 1, dtype=torch.float32)
# x的索引是0到999
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(X=time, Y=[x], xlabel='time', ylabel='x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 将序列映射为模型的特征-标签数据对,特征Xt=[x{t-tau},...,x{t-1}],标签yt=xt
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 一个简单的多层感知机
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

        # # 使用Xavier初始化线性层的权重,旨在保持每层的梯度在传播过程中具有相似的尺度
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 回归问题使用平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

# 模型训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = MyNet()
train(net, train_iter, loss, 5, 0.01)

# 模型预测
# 单步预测，给定原有的4个数据预测下一个数据
# 只预测x[4:999],x[999]基于x[995],x[996],x[997],x[998]
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.show()

# 多步预测
#
multistep_preds = torch.zeros(T)
# 前n_train + tau的数据都是原始数据
multistep_preds[: n_train + tau] = x[: n_train + tau]
# 第n_train + tau + 1到最后一个数据都是模型预测的数据
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()
# 结果发现使用自己的预测进行多步预测时,预测结果很快就衰减到一个常数,是因为错误的累积,导致误差相当快地偏离真实的观测结果。

