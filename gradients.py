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
model=torch.load('./result/best_model.pkl')
model1=torch.load('./result/best_model.pkl')
model2=torch.load('./result/best_model.pkl')
model.eval()
def gradient(x,y,timeindex,num_steps):
    grad=torch.zeros([8,10,19,18])
    test_c = nn.Parameter(Test_c[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    test_p = nn.Parameter(Test_p[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    test_t = nn.Parameter(Test_t[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    optimizer = torch.optim.Adam(list(model.parameters()) + [test_c, test_p, test_t])
    optimizer1 = torch.optim.Adam(list(model1.parameters()) + [test_c, test_p, test_t])
    guide_model = GuidedBackpropReLUModel(model1)
    ouputs=model(test_c,test_p,test_t,-1)
    guide_output=guide_model(test_c,test_p,test_t,-1)
    test_c_intergrated = nn.Parameter(Intergrated_grad(test_c,num_steps))
    test_p_intergrated = nn.Parameter(test_p.repeat(num_steps+1,1,1,1))
    test_t_intergrated = nn.Parameter(test_t.repeat(num_steps+1,1,1,1))
    optimizer2 = torch.optim.Adam(list(model.parameters()) + [test_c_intergrated, test_p_intergrated, test_t_intergrated])
    optimizer3 = torch.optim.Adam(list(model2.parameters()) + [test_c_intergrated, test_p_intergrated, test_t_intergrated])
    guide_model1 = GuidedBackpropReLUModel(model2)
    Intergrated_out=model(test_c_intergrated,test_p_intergrated,test_t_intergrated,-1)
    Intergrated_guidebp_out=guide_model1(test_c_intergrated, test_p_intergrated, test_t_intergrated, -1)
    for i in range(nb_channel):
        grad1 = torch.zeros(num_steps + 1, 10, 19, 18)
        grad2 = torch.zeros(num_steps + 1, 10, 19, 18)
        optimizer.zero_grad()
        ouputs[0,i,x,y].backward(retain_graph=True)
        grad[i]=test_c.grad.cpu()
        optimizer1.zero_grad()
        guide_output[0,i,x,y].backward(retain_graph=True)
        grad[i+2]=test_c.grad.cpu()
        for j in range(num_steps+1):
            optimizer2.zero_grad()
            Intergrated_out[j,i,x,y].backward(retain_graph=True)
            grad1[j]=test_c_intergrated.grad[j].cpu()
            optimizer3.zero_grad()
            Intergrated_guidebp_out[j,i,x,y].backward(retain_graph=True)
            grad2[j]=test_c_intergrated.grad[j].cpu()
        grad1 = (grad1[:-1] + grad1[1:]) / 2.0
        avg_grad = torch.mean(grad1, dim=0)
        grad[i+4] = avg_grad * Test_c[timeindex].unsqueeze(0)
        grad2 = (grad2[:-1] + grad2[1:]) / 2.0
        avg_grad = torch.mean(grad2, dim=0)
        grad[i+6] = avg_grad * Test_c[timeindex].unsqueeze(0)

    #ablation
    test_c = Test_c[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_p = Test_p[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_t = Test_t[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_p =test_p.repeat(num_steps+1,1,1,1)
    test_p =test_t.repeat(num_steps+1,1,1,1)
    ablation_out=torch.zeros([2,19,18])
    for i in range(19):
        for j in range(18):
            test_c_=Ablation(test_c,num_steps,i,j)
            ablation_ouputs=model(test_c_,test_p,test_t,-1)
            ablation_ouputs=torch.mean(ablation_ouputs,dim=0)
            ablation_out[:,i,j]=(ablation_ouputs[:,x,y]-ouputs[0,:,x,y]).squeeze(0)
    return grad.numpy(),ablation_out.detach().numpy()

activation = []
def get_activation(name):
    def hook(model, input, output):
        activation.append(output.detach().cpu().numpy())
    return hook
def hook(timeindex):

    layer_name = 'FusionLayer'
    test_c = Test_c[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_p = Test_p[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    test_t = Test_t[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda()
    model.inputs_c.register_forward_hook(get_activation(layer_name))
    model.inputs_p.register_forward_hook(get_activation(layer_name))
    model.inputs_t.register_forward_hook(get_activation(layer_name))
    out=model(test_c,test_p,test_t,-1)
    return np.squeeze(np.array(activation))

if __name__=='__main__':
    res=hook(timeindex=48)
    np.save('./result/w_output',res)







