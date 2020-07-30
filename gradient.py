from params import *
import  numpy as np
from process import *
from ST_resnet import *
import torch
import  torch.utils.data as Data
from tqdm import tqdm
from torchsummary import summary
from Guide_BP import *
# data dim
c_dim = (nb_channel* len_c,HEIGHT, WIDTH)
p_dim = (nb_channel*len_p,HEIGHT, WIDTH)
t_dim = (nb_channel*len_t,HEIGHT, WIDTH)

# load data
TrainX, TrainY, TestX, TestY = process_data()
Test_c, Test_p, Test_t= (torch.from_numpy(TrainX[0])
                          , torch.from_numpy(TrainX[1]), torch.from_numpy(TrainX[2]))
#load model
model=torch.load('./result/best_model.pkl').train()
def gradient(timeindex):
    test_c = nn.Parameter(Test_c[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    test_p = nn.Parameter(Test_p[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    test_t = nn.Parameter(Test_t[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    optimizer = torch.optim.Adam(list(model.parameters()) + [test_c, test_p, test_t])
    ouputs=model(test_c,test_p,test_t,-1)
    grad=torch.zeros([2,19,18,10,19,18])
    for i in range(nb_channel):
        for j in range(19):
            for k in range(18):
                optimizer.zero_grad()
                ouputs[0,i,j,k].backward(retain_graph=True)
                grad[i,j,k]=test_c.grad.cpu()
    return grad.numpy()


if __name__=='__main__':
    res=gradient(timeindex=48)
    np.save('./result/gradint',res)
    res=np.load('./result/gradint.npy')
    res=np.concatenate(res[np.newaxis],axis=0)
    inflow,outflow=res[0],res[1]
    inflow=inflow.reshape(19*18,10,19*18)
    outflow=outflow.reshape(19*18, 10, 19 * 18)
    outflow






