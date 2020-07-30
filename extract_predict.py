from params import *
import  numpy as np
from process import *
import torch
# data dim
len__c, len__p, len__t = 1, 1, 1
c_dim = (nb_channel*len__c,HEIGHT, WIDTH)
p_dim = (nb_channel*len__p,HEIGHT, WIDTH)
t_dim = (nb_channel*len__t,HEIGHT, WIDTH)
# load data
TrainX, TrainY, TestX, TestY = process_data()
TrainX[0]=np.load('data/input_c_next.npy')[0:144*15].reshape(-1,2,19,18)
model = torch.load('./result/best_model.pkl').to('cuda')
Train_c, Train_p, Train_t = (torch.from_numpy(TrainX[0]),torch.from_numpy(TrainX[1]), torch.from_numpy(TrainX[2]))
Train_c = Train_c.type(torch.FloatTensor).to('cuda')
Train_p = Train_p.type(torch.FloatTensor).to('cuda')
Train_t = Train_t.type(torch.FloatTensor).to('cuda')

out=model(Train_c[4:],Train_p[4:],Train_t[4:],-1)
out=out.detach().cpu().numpy()
out=np.concatenate([Train_c[:4].detach().cpu().numpy(),out],axis=0)
np.save('./result/predict',out)