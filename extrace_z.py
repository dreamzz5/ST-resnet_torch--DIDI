from params import *
import  numpy as np
from process import *
import torch
from torchsummary import summary
from Guide_BP import *
import pickle
# data dim
c_dim = (nb_channel*len_c,HEIGHT, WIDTH)
p_dim = (nb_channel*len_p,HEIGHT, WIDTH)
t_dim = (nb_channel*len_t,HEIGHT, WIDTH)

# load data
TrainX, TrainY, TestX, TestY = process_data()
Test_c, Test_p, Test_t= (torch.from_numpy(TrainX[0])
                          , torch.from_numpy(TrainX[1]), torch.from_numpy(TrainX[2]))
activation = []
def get_activation(name):
    def hook(model, input, output):
        z = output.view(19, 18).unsqueeze(0)
        activation.append(z.detach().cpu().numpy())
    return hook
def hook_z(timeindex,model):
    test_c = Test_c[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_p = Test_p[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_t = Test_t[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    model.linear1.register_forward_hook(get_activation('linear1'))
    model.linear2.register_forward_hook(get_activation('linear2'))
    weight.append(np.squeeze(model.linear1.weight.detach().cpu().numpy()))
    weight.append(np.squeeze(model.linear2.weight.detach().cpu().numpy()))
    np.concatenate(weight,axis=0)
    np.save('./result/weight',weight)
    out=model(test_c,test_p,test_t,-1)
    activation.append(test_c.squeeze(0).detach().cpu().numpy())
    activation.append(Test_c[timeindex+1][-3:-1].numpy())
    return np.squeeze(np.array(activation))
if __name__=='__main__':
    center=[]
    model=torch.load('./result/best_model.pkl').to('cuda')
    kmeans_inflow=pickle.load(open("./result/kmeans_inflow.pkl", "rb"))
    center.append(kmeans_inflow.cluster_centers_[np.newaxis,:])
    kmeans_outflow=pickle.load(open("./result/kmeans_outflow.pkl", "rb"))
    center.append(kmeans_outflow.cluster_centers_[np.newaxis,:])
    center=np.concatenate(center,axis=0)
    np.save('./result/center',center)
    z=hook_z(timeindex=6*12,model=model)
    z=np.array(np.concatenate(z,axis=0))
    np.save('./result/z_out',z)

