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
#print(model)
features = []
loss_fn = nn.MSELoss()
#outputs=outputs*MAX_FLOWIO
grad=np.zeros([2,19,18,10,19,18])
values=np.zeros([2,19,18])
for v in [239]:
    test_c = nn.Parameter(Test_c[v].unsqueeze(0)).cuda()
    test_p = nn.Parameter(Test_p[v].unsqueeze(0)).cuda()
    test_t = nn.Parameter(Test_t[v].unsqueeze(0)).cuda()
    outputs = guide_model(test_c, test_p, test_t, -1)
    for channel in range(1):
        for i in range(0,10):
            grad_1 = np.zeros([10, 19, 18])
            for j in range(0,5):
                outputs[0,channel, i, j].backward(retain_graph=True)
                grad_0=test_c.grad.cpu().numpy()[0]
                optimizer.zero_grad()
                grad_1+=grad_0
                grad[channel,i,j]=grad_1
                if np.max(grad_0)>0:
                    values[channel, i, j] =1
                    print(1)
print(np.max(grad))
np.save('./guide_BP_result/Test_c_grad',grad)
#Intergrated grad
#params
num_steps=49
grad=torch.zeros([num_steps+1,10,19,18])
times=239
for v in [times]:
    test_c=Intergrated_grad(Test_c[v].unsqueeze(0),num_steps=num_steps)
    test_p=Test_p[v].repeat(num_steps+1,1,1,1)
    test_t=Test_t[v].repeat(num_steps+1,1,1,1)
    test_c = nn.Parameter(test_c).cuda()
    test_p = nn.Parameter(test_p).cuda()
    test_t = nn.Parameter(test_t).cuda()
    outputs = guide_model(test_c, test_p, test_t, -1)
    for channel in range(1):
        for i in range(5,6):
            grad_1 = np.zeros([10, 19, 18])
            for j in range(3,4):
                for q in range(num_steps+1):
                    outputs[q,channel, i, j].backward(retain_graph=True)
                    grad[q]=test_c.grad[0]
                    optimizer.zero_grad()
#Intergrated
grad=(grad[:-1]+grad[1:])/2.0
avg_grad=torch.mean(grad,dim=0)
Intergrated_grad=avg_grad*Test_c[times].cpu()
np.save("./guide_BP_result/Intergrated_grad_guide_bp", Intergrated_grad)


