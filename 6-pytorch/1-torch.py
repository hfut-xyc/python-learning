import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# nn.Transformer()

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

'''
view reshape
'''
x = torch.ones(2, 3)
# print(x.data_ptr() == x.reshape(1, 6).data_ptr())
# print(x.data_ptr() == x.transpose(0, 1).reshape(1, 6).data_ptr())

print(x.view(1, 6).is_contiguous())
print(x.reshape(1, 6).is_contiguous())
print(x.flatten().is_contiguous())

print(x.transpose(1, 0).is_contiguous())
print(x.permute(1, 0).is_contiguous())

'''
cat stack
'''
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# print(torch.cat((a, b), 0))       # torch.cat((a, b), 1) error
# print(torch.stack((a, b), 0))
# print(torch.stack((a, b), 1))
# print(torch.hstack((a, b)))
# print(torch.vstack((a, b)))

# x = torch.ones(2, 3)
# y = torch.ones(2, 3) * 2
# print(torch.cat((x, y), 0))
# print(torch.cat((x, y), 1))
# print(torch.stack((x, y), 0))
# print(torch.stack((x, y), 1))
# print(torch.hstack((x, y)))
# print(torch.vstack((x, y)))

# a = torch.tensor([[1],[2],[3]])
# b = torch.tensor([[4],[5],[6]])
# torch.hstack((a,b))

# x = torch.rand(1, 2, 3)
# y = torch.rand(1, 2, 3)
# z = torch.cat((x, y), 1)
# print(z.shape)

'''
expand repeat
'''
# x = torch.tensor([1, 2, 3])
# y = x.expand(2, 3)
# z = x.repeat(2, 3)
# print(x.data_ptr() == y.data_ptr())
# print(x.data_ptr() == z.data_ptr())
# print(y)
# print(z)
# y[1, 2] = 0
# print(x)

'''
transpose permute
'''
# x = torch.rand(2, 3, 4)
# y = torch.transpose(x, 1, 2)
# print(y.shape)
# z = torch.permute(x, (2, 1, 0))
# print(z.shape)

# x = torch.ones(2, 3)
# y = x.view(-1, 2)

'''
squeeze unsqueeze
'''
# a = torch.tensor([[1], [2], [3], [4]])
# print(a.squeeze())
# print(a[:, 1])
# print(a[1, :])
# b = torch.unsqueeze(a, 1)
# c = torch.unsqueeze(a, 2)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# print(b)
# print(c)

'''
math
'''
# x = torch.rand(4, 4)
# print(x)
# print(torch.max(x, 0))
# print(torch.max(x, 1))

# x = torch.rand(4, 4)
# print(x)
# print(x.topk(1))

# x = torch.tensor([
#     [1, 1, 0],
#     [0, 1, 1]
# ])
# print(torch.nonzero(x, as_tuple=True))