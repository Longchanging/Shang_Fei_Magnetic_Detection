# coding:utf-8
'''
@time:    Created on  2018-09-20 14:28:34
@author:  Lanqing
@Func:    Try plot in real time
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    graph_data = open('file.txt', 'r').read() 
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()  # clear everything
    ax1.plot(xs, ys)
    
ani = animation.FuncAnimation(fig, animate, interval=1000)  # one second
plt.show()