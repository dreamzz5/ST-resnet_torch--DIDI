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
c_student_dim=(1* nb_channel+1,HEIGHT, WIDTH)
c_dim = (nb_channel* len_c,HEIGHT, WIDTH)
p_dim = (nb_channel*len_p,HEIGHT, WIDTH)
t_dim = (nb_channel*len_t,HEIGHT, WIDTH)

# load data
TrainX, TrainY, TestX, TestY = process_data()
#np.save('./Z_result/Train_c',TrainX[0])
# to torch
teacher_train_c=torch.from_numpy(TrainX[0])
Train_c, Train_p, Train_t = (torch.from_numpy(TrainX[0][:,:2,:,:])
                             , torch.from_numpy(TrainX[1]), torch.from_numpy(TrainX[2]))
Z=nn.Parameter(torch.zeros([Train_c.shape[0],1,19,18]).double())
Train_c=torch.cat((Train_c,Z),dim=1)
# to cuda
teacher_train_c = teacher_train_c.type(torch.FloatTensor).to('cuda')
Train_c = Train_c.type(torch.FloatTensor).to('cuda')
Train_p = Train_p.type(torch.FloatTensor).to('cuda')
Train_t = Train_t.type(torch.FloatTensor).to('cuda')
# load model
model_teacher=ST_resnet(c_dim,p_dim,t_dim,residual_units=2,day_dim=-1)
model_teacher=torch.load('./result/best_model.pkl')
model_teacher.eval().to('cuda')
#model=ST_resnet(c_dim,p_dim,t_dim,residual_units=2,day_dim=-1).to('cuda')
model=torch.load('./result/best_model.pkl')

def recursive_relu_apply(module_top):
    for idx, module in module_top._modules.items():
        recursive_relu_apply(module)
        if module.__class__.__name__ == 'Conv2d' and module.in_channels==8:
            module_top._modules[idx] = nn.Conv2d(1, 1, kernel_size=3,stride=1, padding=1, bias= True)
            break
def print_grad(a,b,c):
    print('\n', a)
    print('\n', b)
    print('\n', c)
#recursive_relu_apply(model)
model.__init__(c_student_dim,p_dim,t_dim,residual_units=2,day_dim=-1)
model.conv2=nn.Conv2d(1, 32, kernel_size=3,stride=1, padding=1, bias= True)
model.to('cuda')
#summary(model,[(3,19,18),(2,19,18),(2,19,18),(1,1,57)])
#train
loss_fn = nn.MSELoss().to('cuda')
optimizer = torch.optim.Adam(list(model.conv2.parameters()))
optimizer2 = torch.optim.Adam([Z],lr=1e-3)
# params
epcho = 200
batchsize = 100
datasize = Train_c.shape[0]
len = int(datasize / batchsize) + 1
last_loss = 100
for epoch in range(epcho):
    batch = 0
    index = torch.randperm(len)
    train_loss=0
    for i in index:
        model.train()
        start = i * batchsize
        end = ((i + 1) * batchsize if (i + 1) * batchsize < datasize else datasize)
        outputs = model(Train_c[start:end], Train_p[start:end], Train_t[start:end], -1)
        teacher_out = model_teacher(teacher_train_c[start:end], Train_p[start:end], Train_t[start:end], -1)
        teacher_out.detach()
        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss = loss_fn(outputs, teacher_out)
        loss.backward()
        #Z.grad=torch.min(Z.grad,torch.zeros_like(Z.grad))
        #Z.register_hook(print)
        optimizer.step()
        optimizer2.step()
        batch += 1
        train_loss += loss.item()
        print("\r", "Epoch: %d || batch:%d train_loss: %.10f"
              % (epoch + 1, batch, loss.item()), end='', flush=True)
    print('\n')
    if last_loss > train_loss:
        last_loss = train_loss
        torch.save(model.state_dict(), './Z_result/best_model.pkl')
        np.save('./Z_result/Z_output_zeros_decay',Z.cpu().detach().numpy())

