# coding:utf-8
'''
@time:    Created on  2018-09-25 15:44:22
@author:  Lanqing
@Func:    dataExplore.src.real_time_2
'''

import time, random
import math
from collections import deque
from matplotlib import pyplot as plt
import numpy as np
import numpy.fft as fft
plt.style.use('dark_background')
cmap = plt.cm.jet  # winter
start = time.time()

class RealtimePlot:
    def __init__(self, axes, max_entries=1000):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.axes = axes
        self.max_entries = max_entries
        
        # self.axes.set_facecolor('xkcd:mint blue')
        self.lineplot, = axes.plot([], [], "w-")
        self.axes.set_autoscaley_on(True)

    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)
        self.lineplot.set_data(self.axis_x, self.axis_y)
        self.axes.set_xlim(self.axis_x[0], self.axis_x[-1] + 1e-15)
        self.axes.relim(); self.axes.autoscale_view()  # rescale the y-axis

    def animate(self, figure, callback, interval=50):
        import matplotlib.animation as animation
        def wrapper(frame_index):
            self.add(*callback(frame_index))
            self.axes.relim(); self.axes.autoscale_view()  # rescale the y-axis
            return self.lineplot
        animation.FuncAnimation(figure, wrapper, interval=interval)

def main():

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    axes = plt.subplot(311)
    display = RealtimePlot(axes)
    x, y, z = [np.array([0.0])], [1], 0
    sample_rate = 1000
    import pandas as pd
    df = pd.read_csv(u'C:/Users/jhh/Desktop/商飞/实时/' + 'Box_101Shell_.txt', sep=',', header=None)
    replace_real_data = df.values[1:1000, :-1]
    replace_real_data = replace_real_data.reshape([replace_real_data.shape[0] * replace_real_data.shape[1], 1])
    resample_loc = range(int(replace_real_data.shape[0] / sample_rate))  # Resample
    replace_real_data = replace_real_data[resample_loc, :]
    counter_lines = 0
    while True:
        a = replace_real_data[counter_lines]  # random.random() * 100 
        b = time.time() - start
        counter_lines += 1
        x.append(a)
        y.append(b)
        z = x
        if len(x) >= 20:
            z = x[-20:]
        display.add(b, a)
        c = np.abs(fft.fft(z))
        time_step = 1 / 10
        freqs = np.fft.fftfreq(c.size, time_step)
        idx = np.argsort(freqs)
        # plt.subplot(312)
        # plt.plot(freqs[idx], c[idx])
        ax = plt.subplot(313)
        if len(x) == 20:
            plt.specgram(np.array(x)[:, 0], NFFT=20, Fs=10, cmap=cmap, hold=False, noverlap=1)
            ax.cla()  
        plt.pause(0.00001)

if __name__ == "__main__": 
    main()
