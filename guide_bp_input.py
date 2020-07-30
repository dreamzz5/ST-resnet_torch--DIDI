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
# load test data
# use train data
Test_c, Test_p, Test_t = (torch.from_numpy(TrainX[0][1:720])
                          , torch.from_numpy(TrainX[1][1:720]), torch.from_numpy(TrainX[2][1:720]))
TestY = torch.from_numpy(TrainY)
# to cuda
Test_c = Test_c.type(torch.FloatTensor).to('cuda')
Test_p = Test_p.type(torch.FloatTensor).to('cuda')
Test_t = Test_t.type(torch.FloatTensor).to('cuda')
TestY = TestY.type(torch.FloatTensor).to('cuda')
# load model
model=ST_resnet(c_dim,p_dim,t_dim,residual_units=2,day_dim=-1).to('cuda')
model=torch.load('./result/best_model.pkl')
optimizer = torch.optim.Adam(list(model.parameters()) + [Test_c,Test_p,Test_t])
optimizer.zero_grad()
guide_model=GuidedBackpropReLUModel(model)
for name,parameters in model.named_parameters():
    print(name,':',parameters.size())
features = []
loss_fn = nn.MSELoss()
grad=np.zeros([2,19,18,10,19,18])
values=np.zeros([2,19,18])
times=239
for v in [times]:
    Test_c_grad=np.load('./guide_BP_result/Test_c_grad.npy')[0,1,4]
    Test_c_grad[:,1,4]=10
    Test_c_grad=torch.from_numpy(Test_c_grad).type(torch.FloatTensor)
    Test_c_grad=torch.max(Test_c_grad,torch.zeros_like(Test_c_grad))
    Test_c_grad=Test_c_grad/torch.max(Test_c_grad)
    Test_c_grad = nn.Parameter(Test_c_grad.unsqueeze(0).cuda())
    test_p = nn.Parameter(Test_p[v].unsqueeze(0).cuda())
    test_t = nn.Parameter(Test_t[v].unsqueeze(0).cuda())
    outputs = guide_model(Test_c_grad, test_p, test_t, -1)
    np.save('./guide_BP_result/guide_bp_outputs_41_10',outputs.squeeze(0).cpu().detach().numpy())


