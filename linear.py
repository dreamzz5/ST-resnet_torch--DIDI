from params import *
import  numpy as np
from process import *
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt
# load data
TrainX, TrainY, TestX, TestY = process_data()
input_c=np.concatenate([TrainX[0],TestX[0]],axis=0)
np.save('./result/Train_c.npy',input_c)
lens=int(input_c.shape[0])
len_=nb_channel* len_c
inflow,outflow=[],[]
for i in range(lens):
    inflow.append(np.transpose(input_c[i, 0:int(len_)-1:2].reshape(1, int(len_ / 2), 19 * 18),(0,2,1)))
    outflow.append(np.transpose(input_c[i, 1:int(len_):2].reshape(1, int(len_ / 2), 19 * 18),(0,2,1)))
inflow=np.concatenate(inflow,axis=0)
outflow=np.concatenate(outflow,axis=0)
x=np.array([x for x in range(5)]).reshape(-1, 1)
x_next=np.array([5]).reshape(-1, 1)
input_c_next=np.zeros([lens,2,19*18,1])
input_c_next[0:5]=TrainX[0][4].reshape(-1,2,19*18,1)
for i in range(5,145):
    for j in range(19*18):
        linear1=LinearRegression().fit(x,inflow[i,j])
        input_c_next[i,0,j]=linear1.predict(x_next)
        linear2=LinearRegression().fit(x,outflow[i,j])
        input_c_next[i,1,j]=linear2.predict(x_next)
    print(i)
np.save('./data/input_c_next',input_c_next)