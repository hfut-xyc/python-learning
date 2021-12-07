import matplotlib.pyplot as plt

x = [0, 0.1, 0.1, 0.15, 0.15, 0.4, 0.4, 1]
y1 = [1, 1, 0.65, 0.65, 0.45, 0.45, 0.3, 0.3]
y2 = [1, 1, 0.5, 0.65, 0.25, 0.45, 0.15, 0.3]
plt.plot(x, y2)
plt.plot(x, y1, '--', color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend(["Orignal","Interpolate"])
plt.show()