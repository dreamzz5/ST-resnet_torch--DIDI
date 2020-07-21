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
# Test_c=nn.Parameter(Test_c)
# Test_p=nn.Parameter(Test_p)
# Test_t=nn.Parameter(Test_t)
# load model
model=ST_resnet(c_dim,p_dim,t_dim,residual_units=2,day_dim=-1).to('cuda')
model.load_state_dict(torch.load('./result/best_model.pkl'))
optimizer = torch.optim.Adam(list(model.parameters()) + [Test_c,Test_p,Test_t])
optimizer.zero_grad()
guide_model=GuidedBackpropReLUModel(model)
#print(model)
features = []
loss_fn = nn.MSELoss()
#outputs=outputs*MAX_FLOWIO
grad=np.zeros([2,19,18,6,19,18])
values=np.zeros([2,19,18])
for v in [239]:
    Test_c = nn.Parameter(Test_c[v].unsqueeze(0))
    Test_p = nn.Parameter(Test_p[v].unsqueeze(0))
    Test_t = nn.Parameter(Test_t[v].unsqueeze(0))
    outputs = guide_model(Test_c, Test_p, Test_t, -1)
    for channel in range(1):
        for i in range(0,10):
            grad_1 = np.zeros([6, 19, 18])
            for j in range(0,5):
                outputs[0,channel, i, j].backward(retain_graph=True)
                grad_0=Test_c.grad.cpu().numpy()[0]
                optimizer.zero_grad()
                grad_1+=grad_0
                grad[channel,i,j]=grad_1
                if np.max(grad_0)>0:
                    values[channel, i, j] =1
                    print(1)
# output=outputs[0:100, 0, 5, 5].cpu().detach().numpy()
# Testy=TestY[0:100, 0, 5, 5].cpu().numpy()
# loss=np.sum(output-Testy)**2
# print((loss*MAX_FLOWIO**2)/100)
# grad=np.stack(grad)
print(np.max(grad))
np.save("./guide_BP_result/Test_c_grad", grad)
np.save("./guide_BP_result/values", values)
c=np.concatenate((TestY[240].cpu().detach().numpy(),Test_c.cpu().detach().numpy().squeeze(0)),axis=0)
np.save("./guide_BP_result/input_Test_c", c)


