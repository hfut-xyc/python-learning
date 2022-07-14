import torch
import torch.nn as nn
from torch import einsum
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

'''cuda'''
print(torch.cuda.is_available())
print(torch.cuda.device_count())

'''triu tril'''
up = torch.triu(torch.ones(4, 4))
low = torch.tril(torch.ones(4, 4))
print(up)
print(low)


'''view reshape '''
x = torch.ones(2, 3)
print(x.data_ptr() == x.reshape(1, 6).data_ptr())   # True
print(x.data_ptr() == x.transpose(0, 1).reshape(1, 6).data_ptr())   # False

print(x.view(1, 6).is_contiguous())         # True
print(x.reshape(1, 6).is_contiguous())      # True
print(x.flatten().is_contiguous())          # True
print(x.transpose(1, 0).is_contiguous())    # False
print(x.permute(1, 0).is_contiguous())      # False


'''flatten'''
x = torch.randn(4, 3, 10, 10)
y = torch.flatten(x, start_dim=0, end_dim=1)
print(y.shape)

'''chunk'''
x = torch.rand(8, 3, 2)
x = x.chunk(4, dim=0)           # 4 * [2, 3, 2]
x = [temp[None] for temp in x]  # 4 * [1, 2, 3, 2]
x = torch.cat(x, dim=0)
print(x.shape)


'''cat stack hstack vstack'''
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.cat((a, b), 0))       
print(torch.hstack((a, b)))
print(torch.vstack((a, b)))
print(torch.stack((a, b), 0))
print(torch.stack((a, b), 1))

a = torch.tensor([[1],[2],[3]])
b = torch.tensor([[4],[5],[6]])
print(torch.hstack((a, b)))
print(torch.vstack((a, b)))

x = torch.ones(2, 3)
y = torch.ones(2, 3) * 2
print(torch.cat((x, y), 0))
print(torch.cat((x, y), 1))
print(torch.stack((x, y), 0))
print(torch.stack((x, y), 1))
print(torch.hstack((x, y)))
print(torch.vstack((x, y)))


x = torch.rand(3, 4, 5)

print("----mean----")
print(x.mean(dim=1))

print("----max----")
print(torch.argmax(x, 1))
print(torch.max(x, 1))

print("----topk----")
print(torch.topk(x, dim=1, k=3))

print("----sort----")
print(torch.sort(x, dim=1))

print("----nonzero----")
x = torch.tensor([1, 1, 0])
print(torch.nonzero(x))

'''expand repeat'''
x = torch.tensor([1, 2, 3])

y = x.expand(2, 3)
print(y)
print(x.data_ptr() == y.data_ptr())
y[1, 2] = 0
print(x)

z = x.repeat(2, 3)
print(z)
print(x.data_ptr() == z.data_ptr())


'''mul'''
x = torch.ones(3, 4)
y = torch.ones(3, 1) * 2
print(torch.mul(x, y)) 

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[1, 2], [3, 4]])
print(torch.mul(x, y)) 
print(torch.matmul(x, y))

'''einsum'''
x = torch.ones(3, 2)
y = torch.ones(4, 2) * 2
z = einsum('ik, jk -> ij', x, y)
print(z)

def network():
    '''softmax'''
    softmax = nn.Softmax(dim=1)
    x = torch.randn(3, 3)
    y = softmax(x)
    print(y)

    '''AdpativePool2d'''
    pool = nn.AdaptiveMaxPool2d(1)
    x = torch.rand(1, 10, 9, 9)
    y = pool(x)
    print(y.shape)

    pool = nn.AdaptiveAvgPool2d(1)
    x = torch.rand(1, 10, 9, 9)
    y = pool(x)
    print(y.shape)

    '''AdpativePool3d'''
    pool = nn.AdaptiveMaxPool3d(1)
    x = torch.rand(1, 10, 9, 9, 9)
    y = pool(x)
    print(y.shape)

    pool = nn.AdaptiveAvgPool3d(1)
    x = torch.rand(1, 10, 9, 9, 9)
    y = pool(x)
    print(y.shape)

def pad():
    a = torch.tensor([0, 1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = torch.tensor([7, 8])
    d = torch.tensor([9])
    data = [a, b, c, d]

    data = pad_sequence(data, batch_first=True)
    print(data)

if __name__ == "__main__":
    pad()