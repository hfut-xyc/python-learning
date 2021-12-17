'''
slice
'''
a = [1, 2, 3, 4, 5]
print(a[:-1])       # [1, 2, 3, 4]
print(a[:5:2])      # [1, 3, 5]
print(a[::-1])      # [5, 4, 3, 2, 1]
print(a[3::-1])     # [4, 3, 2, 1]
print(a[3:0:-1])    # [4, 3, 2]

'''
list
'''
# list = []
# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])
# list.extend([x, y])
# print(list)


"""
zip
"""
# data = [(1, 'a'), (2, 'b'), (3, 'c')]
# print(*data)
# l1, l2 = zip(*data)
# print(l1)
# print(l2)

# for x, y in zip(l1, l2):
#     print('{}, {}'.format(x, y))

'''
Counter
'''
# from collections import Counter
# c = Counter()
# c.update(["test", "test", 'me'])
# print(c.items())

'''
tqdm
'''
from tqdm import tqdm
import time

for i in tqdm(range(10)):
    time.sleep(0.5)
    pass