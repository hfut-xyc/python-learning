import time
import numpy as np

class State:
    def __init__(self, data, directionFlag = None, parent = None):
        self.data = data
        self.direction = ['up', 'down', 'left', 'right']
        if directionFlag:
            self.direction.remove(directionFlag)
        self.parent = parent

    def showInfo(self):
        print(self.data)
        print('->')

    def getSubStates(self):
        if not self.direction:
            return []
        subStates = []
        border = len(self.data) - 1
        row, col = np.where(self.data == 0)

        # 空位向上移动，产生新的状态节点，加入到subStates中
        if 'up' in self.direction and row > 0:
            s = self.data.copy()
            s[row, col] = s[row - 1, col]
            s[row - 1, col] = 0
            subStates.append(State(data=s, parent=self, directionFlag='down'))

        # 空位向下移动，产生新的状态节点，加入到subStates中
        if 'down' in self.direction and row < border: 
            s = self.data.copy()
            s[row, col] = s[row+1, col]
            s[row + 1, col] = 0
            subStates.append(State(data=s, parent=self, directionFlag='up'))

        # 空位向左移动，产生新的状态节点，加入到subStates中
        if 'left' in self.direction and col > 0:  # 向左移动
            s = self.data.copy()
            s[row, col] = s[row, col-1]
            s[row, col-1] = 0
            subStates.append(State(data=s, parent=self, directionFlag='right'))

        # 空位向右移动，产生新的状态节点，加入到subStates中
        if self.direction.count('right') and col < border: 
            s = self.data.copy()
            s[row, col] = s[row, col+1]
            s[row, col + 1] = 0
            subStates.append(State(data=s, parent=self, directionFlag='left'))
        return subStates

    def BFS(self):
        openTable = []  # 存放状态的地方
        openTable.append(self)  
        paths = []
        while len(openTable) > 0:
            item = openTable.pop(0)  
            subStates = item.getSubStates()
            # 将子状态添加到openTable中
            if len(subStates) > 0:
                openTable.extend(subStates)
            # 查看子状态中有没有最终状态，如果有则输出搜索路径
            for s in subStates:
                if (s.data == target).all():
                    while s.parent and s.parent != s1:
                        paths.append(s.parent)
                        s = s.parent
                    paths.reverse()
                    return paths
        return None


s1 = State(np.array([[2, 8, 3], [1, 6, 4], [7, 0, 5]]))
target = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
start = time.time()
paths = s1.BFS()
end = time.time()
print("Total steps is %d" % len(paths))
print("time consumed:{}".format(end - start))
# if paths:  
#     for node in paths:
#         node.showInfo()
#     print(target)
    
