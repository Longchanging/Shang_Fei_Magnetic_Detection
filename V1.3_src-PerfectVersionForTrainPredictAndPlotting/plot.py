# coding:utf-8
from config import train_folders, sigma, sensor_sample_rate_now
from functions import divide_files_by_name, vstack_list, hstack_list
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.rc('font', family='Helvetica')
cmap = plt.cm.jet  # winter

######  文件相关设置 ,更改为自动切换数据集
base_input, base_output = '../data/input/', 'C:/Users/jhh/Desktop/History_data/explore/'

linewidth, fontsize, fig_size, use_axis_or_not, all_axis, M = 1, 25, (11, 8), True, 'on', 4  # M定义绘制子图格个数
use_package = 'lanqing0926'
use_gauss = True
plot_sample_rate = 100  # 绘图采样频率
NN = 1000 / (sensor_sample_rate_now / plot_sample_rate)  # 计算相邻点的时间间隔， 很重要的处理和标定逻辑

def interact_With_User():  ######  接收用户参数，决定文件夹等
    import os
    key_ = use_package
    folder = base_input + '/' + str(key_) + '/'
    image_folder = base_output + '/' + str(key_) + '/'
    files_train = train_folders[use_package]
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    return folder, image_folder, files_train

folder, image_folder, files_train = interact_With_User()

def read_a_file(fid):  ######  读单个文件
    from scipy.ndimage import filters
    df = pd.read_csv(fid, sep=',', header=None)
    read_array = df.values[:, :-1]
    # read_array = read_array[:250]
    r, c = read_array.shape
    resample_loc = range(int(r / plot_sample_rate))  # Resample
    read_array = read_array[resample_loc, :]  # / 65536
    print('采样前： %d * %d, 采样后： %d * %d' % (r, c, read_array.shape[0], read_array.shape[1]))
    read_array = read_array.reshape([read_array.shape[0] * read_array.shape[1], 1]).T       
    gaussian_X = filters.gaussian_filter1d(read_array, sigma) if use_gauss  else read_array  
    return  gaussian_X

def rewrite_data_prepare_for_plot():
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
        4. 完成修改，默认读入一个一维的矩阵，降低采样率
    '''
    file_dict = divide_files_by_name(folder, files_train)
    file_array_all = []
    for category in files_train:
        files_list = []
        for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list
            file_array = read_a_file(one_category_single_file)
            files_list.append(file_array)
        file_array_one_category = vstack_list(files_list)
        file_array_all.append(file_array_one_category.T)
    file_array_all = hstack_list(file_array_all)
    if len(file_array_all.shape) > 2:
        file_array_all = file_array_all.reshape([file_array_all.shape[0], file_array_all.shape[1]])
    print('Shape of data', file_array_one_category.shape)
    file_array_all = pd.DataFrame(file_array_all)
    file_array_all.columns = files_train
    return file_array_all

def plot(file_, values, i):  #####  进行数据探索 
    name = file_.split('.')[0]
    if '-' in name:
        name = name.split('-')[1]
    fig = plt.figure(figsize=fig_size)  # (figsize=(100, 20))    
    plt.title('Magnetic signals of %s' % name, fontsize=fontsize)  
    ax = fig.add_subplot(111) 
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(use_axis_or_not)
    frame.axes.get_xaxis().set_visible(use_axis_or_not)
    plt.axis(all_axis)
    fig.set_size_inches(fig_size)  # 18.5, 10.5
    count_line = len(values)  #### 认真标定 x 轴数据
    total_time = int(count_line * NN)  # # 总的时间长度(ms)
    X = np.linspace(0, total_time, count_line) / 1000
    plt.xlabel("Sample Time (second) ", size=fontsize)
    plt.ylabel("Magnetic Signal (uT)", size=fontsize)  
    ax.plot(X, values, 'b-', linewidth=linewidth)
    ax.legend(loc='best')
    plt.savefig(image_folder + '%s%d.png' % (file_, i))
    plt.show()
    print('saved to %s' % (image_folder + file_))
    return

def data_explore(file_all):
    ######  归一化
    print('\n文件最大值: \n', np.max(file_all), '\n文件最小值: \n', np.min(file_all), '\n')
    print('\n全局最大值: ', np.max(np.max(file_all)), '\n全局最小值: ', np.min(np.min(file_all)))
    file_all = np.max(np.max(file_all)) + np.min(np.min(file_all)) - file_all  # ##真正转化为uT的函数， 反向
    # file_all = (file_all - np.min(np.min(file_all))) / (np.max(np.max(file_all)) - np.min(np.min(file_all))) ## 归一化
    print(file_all.describe())
    ######  探索
    print('\n数据探索滤波后结果:\n', file_all.describe())
    file_all.plot()
    plt.show()
    ######  绘图
    for file_ in files_train:
        print(file_)
        if not use_gauss:
            values = file_all[file_]
            plot(file_, values, 1)
            slide = int(len(values) / M)
            print(len(values), slide)
            for i in range(M):
                print(i)
                tmp = values.iloc[i * slide : (i + 1) * slide]
                plot('filtered', tmp, i)
        else:
            values = file_all[file_]
            plot(file_ + 'filted_all', values, 1)

if __name__ == '__main__':
    file_all = rewrite_data_prepare_for_plot()
    data_explore(file_all)
