# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    Read data and Preprocess
'''

import time
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from config import train_keyword, train_folder, train_tmp, sigma, overlap_window, window_length, \
    model_folder, train_data_rate, train_folders, train_info_file, \
    sample_rate, batch_size, units, MAX_NB_VARIABLES, cols
from functions import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, \
    min_max_scaler, one_hot_coding, PCA, \
    train_test_evalation_split, knn_classifier, random_forest_classifier, validatePR, check_model, generate_configs, \
    vstack_list

def single_file_process(array_, category, after_fft_data):
    '''
        1. Process after "vstack" 
        2. receive an file array and corresponding category
        3. return a clean array
    '''    
    final_list = []
    numerical_feature_list = []
    rows, cols = array_.shape
    i = 0
    while(i * overlap_window + window_length < rows):  # split window, attention here
        tmp_window = array_[(i * overlap_window) : (i * overlap_window + window_length)]   
        tmp_window = tmp_window.reshape([window_length * cols, 1])   
        # gauss filter
        tmp_window = gauss_filter(tmp_window, sigma)
        numerical_feature_tmp = tmp_window
        # fft process
        tmp_window = fft_transform(tmp_window)
        final_list.append(tmp_window)
        numerical_feature_list.append(numerical_feature_tmp)
        i += 1
    final_array = np.array(final_list)
    numerical_feature_list = np.array(numerical_feature_list)
    print(final_array.shape)
    final_array = final_array.reshape([final_array.shape[0], final_array.shape[1]])
    numerical_feature_array = numerical_feature_list.reshape([numerical_feature_list.shape[0], numerical_feature_list.shape[1]])
    return final_array, numerical_feature_array

def read__data(input_folder, different_category, percent2read_afterPCA_data, after_fft_data_folder):
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
        4. 完成修改，默认读入一个一维的矩阵，降低采样率
    '''
    file_dict = divide_files_by_name(input_folder, different_category)
    fft_list, num_list, label = [], [], []
    for category in different_category:
        file_array_one_category = np.array([[0]] * cols).T  # Initial, skill here
        for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list
            file_array = read_single_txt_file_new(one_category_single_file)
            file_array_one_category = np.vstack((file_array_one_category, file_array))  
        file_array_one_category = file_array_one_category[1:]  # exclude first line
        fft_feature, num_feature = single_file_process(file_array_one_category, category, after_fft_data_folder)  # 预处理
        tmp_label = [category] * len(fft_feature)
        # generate label part and merge all
        fft_list.append(fft_feature) 
        num_list.append(num_feature) 
        label += tmp_label
    fft_data = vstack_list(fft_list)
    data_feature = vstack_list(num_list)
    fft_data = PCA(fft_data)
    ''' Attention Here, Using FFT Only '''
    # data = np.hstack((data_feature, fft_data))
    data = fft_data
    data = min_max_scaler(data) 
    #### 暂时苟合在一起， 最后处理、划分完训练、预测集再分开； 先FFT，后numberic
    label, _ = one_hot_coding(label, 'train')  # not really 'one-hot',haha
    print('Shape of data,shape of label:', data.shape, label.shape)
    return data, label

def baseline_trainTest(data, label):  
    """   
        Train and evaluate using KNN and RF classifier   
    """
    X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label)
    X = X_train
    y = y_train
    print('All samples shape: ', data.shape)
    file_write = model_folder + 'best_model'
    model_save_file = file_write
    model_save = {}
    train_wt = None
    num_train = X.shape[0]
    is_binary_class = (len(np.unique(y)) == 2)
    # test_classifiers = ['NB','KNN', 'LR', 'RF', 'DT','SVM','GBDT','AdaBoost']
    test_classifiers = ['RF']  # , 'DT','LR']  # , 'GBDT', 'AdaBoost']
    classifiers = {'KNN':knn_classifier,
                   'RF':random_forest_classifier }
    # print ('******************** Data Info *********************')
    scores_Save = []
    model_dict = {}
    accuracy_all_list = []
    for classifier in test_classifiers:
        # print ('******************* %s ********************' % classifier)
        scores = []
        skf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        # print(skf_cv) 
        i = 0
        for train_index, test_index in skf_cv.split(X, y):
            i += 1                                               
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = classifiers[classifier](X_train, y_train, train_wt)
            predict_y = model.predict(X_test) 
            Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_y, y_test) 
            # print(accuracy_all, F1Score)
            # print (' \n Precise: %f \n' % Precise, 'Recall: %f \n' % Recall, 'F1Score: %f \n' % F1Score)  # judge model,get score
            scores.append({'cnt:':i, 'mean-F1-Score':Micro_average, 'accuracy-all':accuracy_all})
            accuracy_all_list.append(accuracy_all)
        Micro_average, accuracyScore = [], []
        for item in scores:
            Micro_average.append(item['mean-F1-Score'])
            accuracyScore.append(item['accuracy-all'])
        Micro_average = np.mean(Micro_average)
        accuracyScore = np.mean(accuracyScore)
        scoresTmp = [accuracy_all, Micro_average]
        # print (' \n accuracy_all: \n', accuracy_all, '\nMicro_average:  \n', Micro_average)  # judge model,get score
        scores_Save.append(scoresTmp)
        model_dict[classifier] = model 
    # print ('******************* End ********************')
    scores_Save = np.array(scores_Save)
    max_score = np.max(scores_Save[:, 1])
    index = np.where(scores_Save == np.max(scores_Save[:, 1]))
    index_model = index[0][0]
    model_name = test_classifiers[index_model]
    # print (' \n Best model: %s \n' % model_name)
    print('Test accuracy: ', max_score)
    joblib.dump(model_dict[model_name], file_write)
    ######## 重新调整，打印混淆矩阵
    model_sort = []
    scores_Save1 = scores_Save * (-1)
    sort_Score1 = np.sort(scores_Save1[:, 1])  # inverse order
    for item  in sort_Score1:
        index = np.where(scores_Save1 == item)
        index = index[0][0] 
        model_sort.append(test_classifiers[index])
    #### 使用全部数据，使用保存的，模型进行实验
    model = model_dict[model_name]       
    predict_y_left = model.predict(X_test_left)  # now do the final test
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_y_left, y_test_left) 
    # print ('\n final test: model: %s, F1-mean: %f,accuracy: %f' % (model_sort[0], Micro_average, accuracy_all))
    s1 = metrics.accuracy_score(y_test_left, predict_y_left)
    f2 = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'traditional_train_test_confusion_matrix.csv', f2.astype(int), delimiter=',', fmt='%d')
    # f1 = metrics.fbeta_score(y_test_left, predict_y_left,beta= 0.5)
    # print ('Not mine: final test: model: %s,\n accuracy: %f' % (model_sort[0], s1),)
    print('Matrix:\n', f2.astype(int))
    return  accuracy_all_list, max_score

def main():
    '''
        Complete code for processing one folder
    '''
    # 第二段程序 需要读用户传的命令是什么（训练、测试、预测、基线、模型）
    data, label = read__data(train_folder, train_keyword, train_data_rate, train_tmp)  #### 读数据
    time0 = time.time()
    accuracy_all_list, max_score = baseline_trainTest(data, label)  #### 训练KNN、RF等传统模型
    time1 = time.time()
    s1 = 0.1  # train_lstm()  #### 训练LSTM模型
    time2 = time.time()
    accuracy_all = 0.2  # train_fcn()  #### 训练 FCN模型
    time3 = time.time()
    check_model()  #### 输出 dict对应的标签
    return accuracy_all_list, (time1 - time0), s1, (time2 - time1), accuracy_all, (time3 - time2)

def control_button(train_folder):
    '''
        Collect training info and define which folder to train
    '''
    # 变更参数
    train_keyword, train_folder, test_folder, predict_folder, train_tmp, test_tmp, predict_tmp, \
    train_tmp_test, model_folder, NB_CLASS = generate_configs(train_folders, train_folder)
    # 存储相关信息
    import os
    train_info_file_ = model_folder + train_info_file 
    if os.path.exists(train_info_file_):
        os.remove(train_info_file_)
    fid = open(train_info_file_, 'a')
    fid.write('Index,dataSet,totalRunTime,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES,\
        knn_acc,rf_acc,time_tr,lstm_acc,time_lstm,fcn_acc,time_fcn')
    fid.write('\n')
    fid.close()
    start__time = time.time()
    accuracy_all_list, t1, s1, t2, accuracy_all, t3 = main()
    end__time = time.time()
    run_time = end__time - start__time
    fid = open(train_info_file_, 'a')    
    str_ = '%s,%.3f,%s,%d,%.2f,%d,%d,%d,%d,' % (train_folder, run_time, NB_CLASS, sample_rate, \
                             train_data_rate, window_length, batch_size, units, MAX_NB_VARIABLES)
    fid.write(str_)
    # fid.write('Index,dataSet,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES')
    metrix = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (accuracy_all_list[0], accuracy_all_list[1], t1, \
                                                np.max(s1), t2, \
                                                np.max(accuracy_all), t3)
    fid.write(metrix)
    fid.write('\n')
    fid.close()
    return

def predict(one_timeWindow_data):
    '''
        Predict Labels Based on Every Single Window Data In Real Time
    '''
    #### Processing 
    cols = 730
    tmp_window = one_timeWindow_data.reshape([window_length * cols, 1])    
    numerical_feature_tmp = gauss_filter(tmp_window, sigma)  # gauss filter
    # fft__window = fft_transform(tmp_window)  # fft process
    fft__window = fft_transform(numerical_feature_tmp)  # fft process
    #### Apply Models
    pca = joblib.load(model_folder + "PCA.m")
    min_max = joblib.load(model_folder + "Min_Max.m")
    fft__window = pca.transform(fft__window.T)
    # fft__window = fft__window.reshape([len(fft__window), 1])
    # print('Hello There', numerical_feature_tmp.shape, fft__window.shape)

    ''' Changes Here '''
    # print(numerical_feature_tmp.shape, fft__window.shape)
    # data_window = np.vstack((numerical_feature_tmp, fft__window.T))
    data_window = fft__window
    
    # print(data_window.shape)
    #     train_max, train_min = float(min_max.data_max_), float(min_max.data_min_)
    #     test_max, test_min = np.max(data_window), np.min(data_window)
    #     all_max = train_max if train_max >= test_max else test_max
    #     all_min = train_min if train_min <= test_min else test_min
    #     data_window = (data_window - all_min) / (all_max - all_min)
    data_window = min_max.transform(data_window)
    #### Predict
    file_write = model_folder + 'best_model'
    machine_learning_model = joblib.load(file_write)
    predict_values = machine_learning_model.predict(data_window)  # ## If necessary just .T  # now do the final test
    # print(predict_values)
    label_encoder = check_model()  #### 输出 dict对应的标签
    class_list = label_encoder.classes_
    # print(class_list[int(predict_values[0])])
    return predict_values, class_list[int(predict_values[0])]

def predict_a_file(file_name):
    data = np.loadtxt(file_name)
    tmp_window = data.reshape([data.shape[0] * data.shape[1], 1])    
    numerical_feature_tmp = gauss_filter(tmp_window, sigma)  # gauss filter
    fft__window = fft_transform(numerical_feature_tmp)  # fft process
    tmp_window = fft__window.reshape([int(data.shape[0] / (window_length * data.shape[1])), window_length * data.shape[1]])    
    #### Apply Models
    pca = joblib.load(model_folder + "PCA.m")
    min_max = joblib.load(model_folder + "Min_Max.m")
    fft__window = pca.transform(fft__window.T)
    data_window = fft__window
    return

if __name__ == '__main__':
    control_button('6')
