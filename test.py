import torch
from torch import nn

n_train = 5
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 3, 1)
y_truth = f(x_test)
n_test = len(x_test)

print("train_set")
print(x_train)
print(y_train)
print("query_set")
print(x_test)

# X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat = torch.matmul(attention_weights, y_train)
#
# print(X_repeat)
# print(attention_weights)
# print(y_hat)

X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
print("tile")
print(X_tile)
print(Y_tile)
print("keys-values")
print(keys)
print(values)


# 小批量矩阵乘法
# X = torch.ones((2, 1, 4))
# print(X)
# Y = torch.ones((2, 4, 6))
# print(Y)
# Z = torch.bmm(X, Y)
# print(Z)

# weights = torch.ones((2, 10)) * 0.1
# print(weights)
# values = torch.arange(20.0).reshape((2, 10))
# print(values)
# output = torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
# print(output)