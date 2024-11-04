import matplotlib.pyplot as plt
import numpy as np

def plot():
    x = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    relu = np.maximum(0, x)
    dist = -0.15

    #################################################
    fig = plt.figure(figsize=(16, 4))
    fig.add_subplot(131)
    plt.plot(x, sigmoid)
    plt.title('(a) Sigmoid', y=dist)
    plt.grid(linestyle='-.')
    
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    #################################################
    fig.add_subplot(132)
    plt.plot(x, tanh)
    plt.title('(b) tanh', y=dist)
    plt.ylim(-1.5, 1.5)
    plt.grid(linestyle='-.')
    
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    #################################################
    fig.add_subplot(133)

    plt.title('(c) ReLU', y=dist)
    plt.plot(x, relu)
    plt.ylim(0, 6)
    plt.grid(linestyle='-.')
    
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')


    # plt.show()
    plt.subplots_adjust(wspace=0.06)

    plt.savefig('test.png', bbox_inches='tight')


if __name__ == '__main__':
    plot()
