import re


'''
greedy
'''
str = '<img src="http://www.image.com/29233315.jpg" alt="alt" title="title">'
match = re.search(r'<img src="(.*)"', str)
print(match.group(0))


'''
lazy
'''
str = '<img src="http://www.image.com/29233315.jpg" alt="alt" title="title">'
match = re.search(r'<img src="(.*?)"', str)
print(match.group(0))


