# coding:utf-8
'''  
    input and preprocess。
    Train 包括完整的train evaluation test, 
    test 指的是完全相同的数据类型代入计算，
    predict指的是没有标签。 
'''

dict_all_parameters = {}
train_info = {}
train_folders = {
            # 'shangfei':['x2', 'x10', 'x30']
            '4':['Arm_102Shell_', 'Arm_103Shell_', 'Arm_104Shell_', 'Box_101Shell_'],
            # '3':['Arm_102X_20_-', 'Arm_102X_20_+', 'Arm_102Y_20_-', 'Arm_102Y_20_+', 'Arm_102Z_20_-', \
            #     'Arm_102Z_20_+'],
            '3':['Box_101X_20_-', 'Box_101X_20_+', 'Box_101Y_20_-', 'Box_101Y_20_+', 'Box_101Z_20_-', \
                'Box_101Z_20_+'],
            # '3':['Arm_104X_20_-', 'Arm_104X_20_+', 'Arm_104Y_20_-', 'Arm_104Y_20_+', 'Arm_104Z_20_-', \
            #  'Arm_104Z_20_+']
            # '3':['Arm_103X_20_-', 'Arm_103X_20_+', 'Arm_103Y_20_-', 'Arm_103Y_20_+', 'Arm_103Z_20_-', \
            #     'Arm_103Z_20_+']
            '5':['Arm_104_3_A2_No', 'Box_101_3_A2_No', 'Box_101_2_A2_30_-', 'Arm_104_2_A2_30_-', 'Arm_104_A2_30_+', 'Box_101_A2_30_+'],
            '6':['Arm_103_3_A2_20_-', 'Arm_103_3_A2_20_+', 'Arm_103_3_A2_30_-', 'Arm_103_3_X_10_-', 'Arm_103_3_X_10_+', 'Box_101_3_A2_30_+_2']
            }

# 采样
sample_rate = 10  # 单位是毫秒 ，>=1
epochs, n_splits = 2 , 10  # 10折交叉验证和epoch数量固定
train_batch_size = 10

# 文件列数
cols = 730
max_entries = int(1000 * 730 / sample_rate)  # 实时绘图窗口大小，单位：绘图窗口总数据包个数，e.g. 200/(4800点/每包730个点) = 30个点

# 处理
saved_dimension_after_pca, sigma = 200 , 5  # default 1000
use_gauss, use_pca, use_fft = True, True, True  # True
whether_shuffle_train_and_test = True
# 训练
test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
batch_size = 10  # [2, 5, 10]  # 训练 batch大小
units = 1  # [20, 10, 50, 200]  # int(MAX_NB_VARIABLES / 2)
# 循环遍历
window_length = 15  # default 50  # [2, 5, 10, 20, 100]  # 窗口大小
train_data_rate = 1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
i = 0
# 参数
overlap_window = int(0.1 * window_length)  # 窗口和滑动大小
MAX_NB_VARIABLES = window_length * 2
MAX_NB_VARIABLES = (window_length + saved_dimension_after_pca) if use_pca else window_length * 2

### 此处添加文件相关信息 ###
train_info_file = 'train_info_all.txt'
train_keyword = ['Arm_103_3_A2_20_-', 'Arm_103_3_A2_20_+', 'Arm_103_3_A2_30_-', 'Arm_103_3_X_10_-', 'Arm_103_3_X_10_+', 'Box_101_3_A2_30_+_2']
train_folder = '../data//input//6/'
test_folder = '../data//input//6/'
predict_folder = '../data//input//6/'
train_tmp = '../data//tmp/6//tmp/train/'
test_tmp = '../data//tmp/6//tmp/test/'
predict_tmp = '../data//tmp/6//tmp/predict/'
train_tmp_test = '../data//tmp/6//tmp/train/test/'
model_folder = '../data//tmp/6//model/'
NB_CLASS = 6
