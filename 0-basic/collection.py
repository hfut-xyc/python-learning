import collections

'''
Counter
''' 
counter = collections.Counter()
counter.update(["test", "test", 'me'])
print(counter.items())

'''
OrderedDict
'''
map = collections.OrderedDict()
map['id'] = 100
map['name'] = 'Tom'
map['birth'] = '1999'
map['address'] = 'xxxx'

for k, v in map.items():
    print(k, v)
