from params import *
import pandas as pd
import datetime as dt
import numpy as np
import torch
def process_data( ):
    data=np.load('./data/all_data_30days_10min.npy')
    #norm
    MAX_FLOWIO=np.max(data)
    data=data/MAX_FLOWIO
    data_range= data.shape[0]
    Startindex,Endindex=0,int(DAYTIMESTEP*22)
    interval_p,interval_t=1,7
    depends = [range(len_c+1,1,-1),
               [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)],
               [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]]

    #train
    start_train = max(len_c, interval_p * DAYTIMESTEP * len_p, interval_t * DAYTIMESTEP * len_t)
    end_train = Endindex
    #test
    start_test = Endindex
    end_test = data_range

    # dayinfo
    if dayInfo:
        day_info = np.genfromtxt('./data/day_information_onehot.csv', delimiter=',', skip_header=1)
        dayInfo_dim = day_info.shape[1]
        train_feature = day_info[start_train:end_train]
        test_feature = day_info[start_test:end_test]
    else:
        train_feature = None
        test_feature =None
        dayInfo_dim = 0

    train_c,train_p,train_t=[],[],[]
    test_c,test_p,test_t=[],[],[]
    for i in range(start_train,end_train):
        train_c.append(np.concatenate([data[i - j] for j in depends[0]],axis=0))
        train_p.append(np.concatenate([data[i - j] for j in depends[1]],axis=0))
        train_t.append(np.concatenate([data[i - j] for j in depends[2]],axis=0))
    for i in range(start_test, end_test):
        test_c.append(np.concatenate([data[i - j] for j in depends[0]] ,axis=0))
        test_p.append(np.concatenate([data[i - j] for j in depends[1]] ,axis=0))
        test_t.append(np.concatenate([data[i - j] for j in depends[2]] ,axis=0))
    train_c, train_p, train_t = np.array(train_c), np.array(train_p), np.array(train_t)
    test_c, test_p, test_t = np.array(test_c), np.array(test_p), np.array(test_t)

    TrainX = [train_c, train_p, train_t, train_feature] if train_feature is not None else [train_c, train_p, train_t]
    TrainY = data[start_train:end_train]
    TestX = [test_c, test_p, test_t, test_feature] if test_feature is not None else [test_c, test_p, test_t]
    TestY =  data[start_test:end_test]
    return TrainX,TrainY,TestX,TestY

if __name__=='__main__':
    process_data()