import numpy as np

'''
slice
'''

a = [1, 2, 3, 4, 5]
# print(a[:-1])       # [1, 2, 3, 4]
# print(a[:5:2])      # [1, 3, 5]
# print(a[::-1])      # [5, 4, 3, 2, 1]
# print(a[3::-1])     # [4, 3, 2, 1]
# print(a[3:0:-1])    # [4, 3, 2]

# a = np.array(a)
# print(a[...])
print(i for i in range(5))
print([i for i in range(5)])

'''
list
'''
# list = [1, 2]
# list.extend([3, 4])
# print(list)

'''
string
'''
# unicode_str = u'我爱中国'
# print(unicode_str)

# raw_str = r'test\n'
# print(raw_str)

# x = 2333
# format_str = f'test{x}'
# print(format_str)

# byte_str = '我爱中国'.encode('utf-8')
# print(byte_str)


'''
zip
'''
# data = [(1, 'a'), (2, 'b'), (3, 'c')]
# print(*data)
# l1, l2 = zip(*data)
# print(l1)
# print(l2)

# for x, y in zip(l1, l2):
#     print('{}, {}'.format(x, y))



'''
tqdm
'''
# from tqdm import tqdm
# import time

# for i in tqdm(range(10)):
#     time.sleep(0.5)
#     pass
