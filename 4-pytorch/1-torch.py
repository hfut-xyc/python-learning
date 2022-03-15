import torch
from torch import einsum

'''
cuda
'''
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

'''
einsum
'''
# x = torch.ones(3, 2)
# y = torch.ones(4, 2) * 2
# z = einsum('ik, jk -> ij', x, y)
# print(z)

'''
chunk
'''
x = torch.rand(16, 3, 2)
x = x.chunk(4, dim=0)           # 4 * [4, 3, 2]
x = [temp[None] for temp in x]  # 4 * [1, 4, 3, 2]
x = torch.cat(x, dim=0)
print(x.shape)  
x = torch.flatten(x, start_dim=0, end_dim=1) 


'''
view reshape 
'''
# x = torch.ones(2, 3)
# print(x.data_ptr() == x.reshape(1, 6).data_ptr())
# print(x.data_ptr() == x.transpose(0, 1).reshape(1, 6).data_ptr())

# print(x.view(1, 6).is_contiguous())     # True
# print(x.reshape(1, 6).is_contiguous())  # True
# print(x.flatten().is_contiguous())      # True
# print(x.transpose(1, 0).is_contiguous())    # False
# print(x.permute(1, 0).is_contiguous())      # False

'''
flatten
'''
# x = torch.randn(4, 3, 10, 10)
# y = torch.flatten(x, start_dim=0, end_dim=1)
# z = x.view(-1, 10, 10)
# print(y.shape)
# print(z.shape)

'''
cat stack hstack vstack
'''
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# print(torch.cat((a, b), 0))       
# print(torch.hstack((a, b)))
# print(torch.vstack((a, b)))
# print(torch.stack((a, b), 0))
# print(torch.stack((a, b), 1))

# a = torch.tensor([[1],[2],[3]])
# b = torch.tensor([[4],[5],[6]])
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
max
'''
# x = torch.rand(4, 4)
# print(x)
# print(torch.max(x, 0))
# print(torch.max(x, 1))

'''
topk
'''
# x = torch.rand(4, 4)
# print(x)
# print(x.topk(1))

'''
nonzero
'''
# x = torch.tensor([
#     [1, 1, 0],
#     [0, 1, 1]
# ])
# print(torch.nonzero(x, as_tuple=True))

'''
mean
'''
# x = torch.ones(3, 4)
# print(x.mean(dim=0))
# print(x.mean(dim=1))


'''
multiply
'''
# x = torch.ones(3, 4)
# y = torch.rand(3, 1)
# print(x * y) # torch.mul(x, y)

# x = torch.tensor([[1, 2], [3, 4]])
# y = torch.tensor([[1, 2], [3, 4]])
# print(x * y) # torch.mul(x, y)