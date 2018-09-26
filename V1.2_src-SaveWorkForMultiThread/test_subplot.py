#!/usr/bin/env python3
 
import time
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

sample_rate = 100  # 采样率，不能太高，否则不好绘图
data_line_read = 200000 
max_entries = 100
plot_pause = 0.00000000001  # second
fft_time_step = 1 / 10
cmap = plt.cm.jet  # winter

def generate_array_data():
    import pandas as pd
    df = pd.read_csv(u'C:/Users/jhh/Desktop/商飞/实时/' + 'Box_101Shell_.txt', sep=',', header=None)
    replace_real_data = df.values[:data_line_read, :-1]
    replace_real_data = replace_real_data.reshape([replace_real_data.shape[0] * replace_real_data.shape[1], 1])
    resample_loc = range(int(replace_real_data.shape[0] / sample_rate))  # Resample
    replace_real_data = replace_real_data[resample_loc, :]
    return replace_real_data

def fft(one_window_data):
    import numpy.fft as fft
    fft_data = np.abs(fft.fft(one_window_data))
    freqs = np.fft.fftfreq(fft_data.size, fft_time_step)
    idx = np.argsort(freqs)
    return freqs[idx], fft_data[idx]

class RealtimePlot:
    def __init__(self, axes, max_entries=max_entries):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.axes = axes
        self.max_entries = max_entries
        
        self.lineplot, = axes.plot([], [], "ro-")
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
    start = time.time()
    fig = plt.figure()
    axes1 = fig.add_subplot(311)
    axes2 = fig.add_subplot(312)
    axes3 = fig.add_subplot(313)
    display1 = RealtimePlot(axes1)
    data, i = generate_array_data(), 0
    Y = []
    while True:
        x, y = time.time() - start, data[i]
        Y.append(y)
        display1.add(x, y)
        fft_x, fft_y = fft(Y[-max_entries:]) 
        index = int(len(fft_x) / 2)
        axes2.plot(fft_x[-index:], fft_y[-index:])
        if len(Y) > 10:
            Z = np.array(Y[-max_entries * 10:])[:, 0]
            f, t, Sxx = signal.spectrogram(Z, fs=1)
            axes3.pcolormesh(t, f, Sxx)
        plt.pause(plot_pause)
        axes2.cla()
        # axes3.cla()
        i += 1
if __name__ == "__main__": main()
