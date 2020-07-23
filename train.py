from params import *
import  numpy as np
from process import *
from ST_resnet import *
import torch
from torch.nn import init
import  torch.utils.data as Data
from tqdm import tqdm
from torch.autograd import Variable
from Guide_BP import *
# data dim
c_dim = (nb_channel* len_c,HEIGHT, WIDTH)
p_dim = (nb_channel*len_p,HEIGHT, WIDTH)
t_dim = (nb_channel*len_t,HEIGHT, WIDTH)

# load data
TrainX, TrainY, TestX, TestY = process_data()

def train(TrainX, TrainY):
    # to torch
    Train_c, Train_p, Train_t = (torch.from_numpy(TrainX[0])
                                 , torch.from_numpy(TrainX[1]), torch.from_numpy(TrainX[2]))
    TrainY = torch.from_numpy(TrainY)
    # to cuda
    Train_c = Variable(Train_c.cuda(), requires_grad=True)
    Train_c = Train_c.type(torch.FloatTensor).to('cuda')
    Train_p = Train_p.type(torch.FloatTensor).to('cuda')
    Train_t = Train_t.type(torch.FloatTensor).to('cuda')
    TrainY = TrainY.type(torch.FloatTensor).to('cuda')
    # validation data
    index = int(Train_c.shape[0] * 0.875)
    Train_c, Validation_c = Train_c[:index], Train_c[index:]
    Train_p, Validation_p = Train_p[:index], Train_p[index:]
    Train_t, Validation_t = Train_t[:index], Train_t[index:]
    TrainY, ValidationY = TrainY[:index], TrainY[index:]
    #load model
    model=ST_resnet(c_dim,p_dim,t_dim,residual_units=2,day_dim=-1)
    loss_fn=nn.MSELoss()
    Train_c=nn.Parameter(Train_c)
    optimizer = torch.optim.Adam(list(model.parameters()),lr=1e-4)
    model.to('cuda')
    loss_fn.to('cuda')
    #params
    epcho=100
    batchsize= 100
    datasize= Train_c.shape[0]
    lens =int(datasize/batchsize)+1
    last_loss=100
    train_loss_=np.zeros([epcho,1])
    val_loss_=np.zeros([epcho,1])
    for epoch in range(epcho):
        batch=0
        index = torch.randperm(lens)
        loss=0
        for i  in index:
            model.train()
            start= i*batchsize
            end  = ((i+1)*batchsize if (i+1)*batchsize<datasize else datasize)
            y=TrainY[start:end]
            outputs = model(Train_c[start:end], Train_p[start:end], Train_t[start:end],-1)
            optimizer.zero_grad()
            loss= loss_fn(outputs, y)
            train_loss_[epoch]+=loss.item()/len(index)
            loss.backward()
            optimizer.step()
            batch+=1
            print("\r","Epoch: %d || batch:%d train_loss: %.10f"
            % (epoch+1, batch, loss.item()), end='', flush=True)

        # valid after each training epoch
        model.eval()
        Validation_output=model(Validation_c,Validation_p,Validation_t,-1)
        val_loss=loss_fn(Validation_output,ValidationY)
        val_loss_[epoch] = val_loss.item()
        if last_loss>val_loss:
            last_loss=val_loss
            torch.save(model, './result/best_model.pkl')
        print('\n'+'*'*40)
        print("Epoch: %d || validation_loss: %.10f"
              % (epoch + 1, val_loss.item()))
        print('*' * 40)
        np.save('./result/train_loss',train_loss_)
        np.save('./result/val_loss', val_loss_)

def test(TestX, TestY):
    # load test data
    Test_c, Test_p, Test_t = (torch.from_numpy(TestX[0])
                              , torch.from_numpy(TestX[1]), torch.from_numpy(TestX[2]))
    TestY = torch.from_numpy(TestY)
    # to cuda
    Test_c = Test_c.type(torch.FloatTensor).to('cuda')
    Test_p = Test_p.type(torch.FloatTensor).to('cuda')
    Test_t = Test_t.type(torch.FloatTensor).to('cuda')
    TestY = TestY.type(torch.FloatTensor).to('cuda')
    # load model
    model = torch.load('./result/best_model.pkl').to('cuda')
    model.eval()
    loss_fn = nn.MSELoss()
    batchsize = 500
    datasize = Test_c.shape[0]
    len = int(datasize / batchsize) + 1
    mse_loss = 0
    n = 0
    out=np.zeros([Test_c.shape[0],2,19,18])
    for i in range(len):
        start = i * batchsize
        end = ((i + 1) * batchsize if (i + 1) * batchsize < datasize else datasize)
        y = TestY[start:end]
        outputs = model(Test_c[start:end], Test_p[start:end], Test_t[start:end], -1)
        out[start:end]=outputs.cpu().detach().numpy()
        loss = loss_fn(outputs[:, :, 5, 3], y[:, :, 5, 3])
        mse_loss += loss.item()
    time_loss=np.sum((1/(19*18))*(TestY.cpu().detach().numpy()-out)**2*MAX_FLOWIO**2,axis=(1,2,3))
    print("Test_loss: %.10f" % (mse_loss / len))
    print("Test_Rescaled_loss: %.10f" % (mse_loss / len * MAX_FLOWIO ** 2))
    # write result
    f = open('./result/prediction_scores.txt', 'a')
    f.seek(0)
    f.truncate()  # 清空文件
    f.write("Keras MSE on test, %f\n" % (mse_loss / len))
    f.write("Rescaled MSE on test, %f\n" % (mse_loss / len * MAX_FLOWIO ** 2))
    f.close()
    return (mse_loss / len * MAX_FLOWIO ** 2),time_loss,out

if __name__ == '__main__':
    #trian
    train(TestX, TestY)
    #test
    _,time_loss,__=test(TestX, TestY)
    np.save('./result/time_loss',time_loss)
    #test grad bp
    add_res=np.zeros([19,18])
    origin_res=np.zeros([19,18])
    x=[TestX[0][239][np.newaxis,:],TestX[1][239][np.newaxis,:],TestX[2][239][np.newaxis,:]]
    y=TestY[239][np.newaxis,:]
    _, _, out = test(x, y)
    origin_res[0:19,0:18] = out[0,0,5,3]
    for i in range(19):
        for j in range(18):
            t=x[0][0,0,i,j]
            x[0][0, 0, i, j]=0
            _,_,out=test(x, y)
            add_res[i,j]=out[0,0,5,3]
            x[0][0, 0, i, j]=t

    np.save('./result/add_grad_loss',add_res)
    np.save('./result/origin_loss', origin_res)
