import torch
import torch.nn as nn

'''
softmax
'''
# softmax = nn.Softmax(dim=1)
# x = torch.randn(3, 3)
# y = softmax(x)
# print(y)


'''
MaxPool
'''
# pool = nn.MaxPool3d((3, 3, 3))
# x = torch.rand(1, 10, 9, 9, 9)
# y = pool(x)
# print(y.shape)

'''
AdpativeAvgPool
'''
pool = nn.AdaptiveAvgPool2d(1)
x = torch.rand(1, 10, 9, 9)
y = pool(x)
print(y.shape)

pool = nn.AdaptiveAvgPool3d(1)
x = torch.rand(1, 10, 9, 9, 9)
y = pool(x)
print(y.shape)