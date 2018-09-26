# coding:utf-8
'''
@time:    Created on  2018-09-11 15:04:53
@author:  Lanqing
@Func:    Real Time Data Collecting, Saving, And Real-Time Plotting, Predicting 
'''
import socket
import numpy as np
from config import window_length, max_entries, sample_rate
from functions import  logistics, plot, RealtimePlot
from main import predict
from matplotlib import pyplot as plt
plt.style.use('dark_background')

t = 0  # 全局变量， 用于实时预测， 即： 取多少次(t)的预测众数作为预测结果
label_list = []  # 全局变量，在t次预测中的全部label列表
data_line_read = 200000 
plot_pause = 0.00000000001  # second
fft_time_step = 1 / 10
cmap = plt.cm.jet  # winter
from scipy import signal

def raw_data_process(content):
    '''
        Process Raw Sensor Data Following Collector Logistics
    '''
    one_package_data = []
    prior_byte = 0b0
    for bin_byte in content:
        if (bin_byte & 0b1) == 0b1:  # if (bin_byte >> 1 << 1) != bin_byte:  # high byte
            prior_byte = bin_byte
        else:  # low byte
            num = (prior_byte >> 1 << 7) | ((bin_byte >> 1) & 0b1111111)
            if num > 8192:
                num -= 16384
            one_package_data.append(num)
    return one_package_data

def reduce_sample_rate(package_list, sample_rate):
    ''' 
        Reduce Sample Rate so that my small PC can process If Necessary
    '''
    package_list = np.array(package_list)
    length_ = len(package_list)
    sample_loc = list(range(0, length_, sample_rate))
    sample_array = package_list[sample_loc]
    return sample_array

def write_one_window_into_file(file_name, D2_list_):
    ''' 
        Write One Window (Window_size * Package_Num) Data Into File 
    '''
    fid = open(file_name, 'a')
    for list_ in D2_list_:
        for item in list_:
            fid.write(str(item))
            fid.write(',')
        fid.write('\n')
    fid.close()
def fft(one_window_data):
    fft_time_step = 0.01
    import numpy.fft as fft
    fft_data = np.abs(fft.fft(one_window_data)) ** 2
    freqs = np.fft.fftfreq(fft_data.size, fft_time_step)
    idx = np.argsort(freqs)
    return freqs[idx], fft_data[idx]

def new_real_time(file_folder, file_names):
    '''
        Real Time Plotting And Saving Data
    '''
    import time
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.bind(("0.0.0.0", 8000))
    start_time = time.time()
    sensor_data, pacakage_counter, file_name = {}, {}, {}
    sensor_data['101'], sensor_data['102'] , sensor_data['103'] , sensor_data['104'] = [], [], [], []
    file_name['101'], file_name['102'] , file_name['103'] , file_name['104'] = 'Box_101', 'Arm_102', 'Arm_103', 'Arm_104'
    pacakage_counter['101'], pacakage_counter['102'] , pacakage_counter['103'] , pacakage_counter['104'] = 0, 0, 0, 0
    collect_time = 50  # collect data of minutes collect a file
    start = time.time()
    fig = plt.figure()
    axes1 = fig.add_subplot(311)  # define figures
    axes2 = fig.add_subplot(312)
    axes3 = fig.add_subplot(313)
    display1 = RealtimePlot(axes1)
    Y, X = [], []
    previous_package_time = start - 1
    while(True):
        content, destInfo = udpSocket.recvfrom(2048)    
        one_package_data = raw_data_process(content)  # get one package data,e.g. 730 data in a package
        address = str(destInfo[0].split('.')[-1])  # sensor address,e.g. 103
        if address and address in sensor_data.keys():  # address in the four sensors
            sensor_data[address].append(one_package_data) 
            time_now, package_data = time.time(), one_package_data  # define x and y axis
            package_data = reduce_sample_rate(package_data, sample_rate)
            time_axis = np.linspace(int(previous_package_time) % 10000, int(time_now) % 10000, len(package_data))
            previous_package_time = time_now
            Y.extend(package_data)
            X.extend(time_axis)
            Y = Y[-max_entries:]
            display1.add(time_axis, package_data)
            fft_x, fft_y = fft(Y)  # (Y[-int(max_entries / 10):]) 
            index = int(len(fft_x) / 2)
            axes2.plot(fft_x[-index:], fft_y[-index:])
            Z = np.array(Y)
            f, t, Sxx = signal.spectrogram(Z, fs=1000)
            axes3.pcolormesh(t, f, Sxx)
            plt.pause(0.0001)
            axes2.cla()            
            pacakage_counter[address] += 1
            if pacakage_counter[address] % window_length == 0:  # every window_size write into file
                write_one_window_into_file(file_folder + file_name[address] + file_names, sensor_data[address])
                sensor_data[address] = []  # clear and restart to collect window_size of data
            if pacakage_counter[address] % 1000 == 0:
                time_now__ = time.time()
                if (time_now__ - start_time) / 60 > collect_time:  # define collect time, e.g. 5 minutes. Can also manually stop
                    udpSocket.close()  # if manually stop, can set longer collect time.
                    break

def real_time_processor():
    import time
    one_time_window_data = []
    pacakage_counter = 0
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.bind(("0.0.0.0", 8000))
    while(True):
        content, destInfo = udpSocket.recvfrom(2048)    
        one_package_data = raw_data_process(content)
        # print('Got')
        # if np.max(np.array(one_package_data)) < 200:
        pacakage_counter += 1
        if pacakage_counter <= window_length:
            one_time_window_data.append(one_package_data)
        else:
            one_window_data = np.array(one_time_window_data).reshape([window_length, len(one_package_data)])
            # print(one_window_data.shape)
            id, label_predict = predict(one_window_data)
            udpSocket.close()
            break
    return one_window_data, id[0], label_predict

def do_post():
    from socket import socket, AF_INET, SOCK_STREAM
    s = socket(AF_INET, SOCK_STREAM)
    ip_port = ('', 8001)
    s.bind(ip_port)
    s.listen(5)
    import time
    while 1:
        conn, addr = s.accept()
        while 1:
            string_post, is_common = real_time()
            if  is_common:
                print('send to client:', string_post)
                conn.sendall(string_post.encode(encoding="utf-8"))
            time.sleep(1)
    return

def real_time():
    global t
    global label_list
    #### 处理结果需要加入自定义逻辑
    is_common = False
    top_list = []
    dict_apps = {}
    dict_apps['NoLoad'] = 0
    dict_apps['safari_surfing'] = 1
    dict_apps['tencent_video'] = 2
    dict_apps['zuma_game'] = 3 
    dict_apps['panhao'] = 5 
    dict_apps['lanqing'] = 4   
    dict_apps['a'] = 5
    dict_apps['b'] = 6
    dict_apps['c'] = 7
    dict_apps['1'] = 8
    dict_apps['2'] = 9
    dict_apps['3'] = 10    
    prior_ones = 'safari_surfing'
    string_post = 'Current Running APP is:0'
    t += 1
    one_window_data, _, label_predict = real_time_processor()
    label_list.append(label_predict)
    print(label_predict)
    if t % 10 == 0:
        from collections import Counter
        word_counts = Counter(label_list)
        top1 = word_counts.most_common(1)[0][0]
        top1, prior_ones = logistics(top1, label_list, prior_ones, top_list)
        print('APP using now:\t', top1)
        prior_ones = top1
        id_ = dict_apps[top1]
        string_post = 'Current Running APP is:%d' % (id_)
        # print(string_post)
        # do_post(string_post)
        top_list.append(top1)
        is_common = True
        label_list = []
    # plot('%d' % t, one_window_data)
    return string_post, is_common


##############          Real Time      ##############
# while True:
#     real_time()

##########       Client Communication         #######
# do_post()

##########   History Collect And Real-Time Plot #######
new_real_time('C:/Users/jhh/Desktop/商飞/实时/', 'Shell_.txt')
