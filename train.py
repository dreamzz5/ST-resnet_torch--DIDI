from params import *
import  numpy as np
from process import *
from ST_resnet import *
import torch
import  torch.utils.data as Data
from tqdm import tqdm
from Guide_BP import *
# data dim
c_dim = (nb_channel* len_c,HEIGHT, WIDTH)
p_dim = (nb_channel*len_p,HEIGHT, WIDTH)
t_dim = (nb_channel*len_t,HEIGHT, WIDTH)



if __name__ == '__main__':
    # load data
    TrainX, TrainY, TestX, TestY = process_data()
    #to torch
    Train_c,Train_p,Train_t = (torch.from_numpy(TrainX[0])
        ,torch.from_numpy(TrainX[1]),torch.from_numpy(TrainX[2]))
    TrainY = torch.from_numpy(TrainY)
    #to cuda
    Train_c   = Train_c.type(torch.FloatTensor).to('cuda')
    Train_p   = Train_p.type(torch.FloatTensor).to('cuda')
    Train_t   = Train_t.type(torch.FloatTensor).to('cuda')
    TrainY    = TrainY.type(torch.FloatTensor).to('cuda')
    #validation data
    index=int(Train_c.shape[0]*0.875)
    Train_c, Validation_c = Train_c[:index], Train_c[index:]
    Train_p, Validation_p = Train_p[:index], Train_p[index:]
    Train_t, Validation_t = Train_t[:index], Train_t[index:]
    TrainY, ValidationY = TrainY[:index], TrainY[index:]
    #load model
    model=ST_resnet(c_dim,p_dim,t_dim,residual_units=2,day_dim=-1)
    loss_fn=nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()))
    model.to('cuda')
    loss_fn.to('cuda')
    #params
    epcho=200
    batchsize= 100
    datasize= Train_c.shape[0]
    len =int(datasize/batchsize)+1
    last_loss=100
    for epoch in range(epcho):
        batch=0
        index = torch.randperm(len)
        for i  in index:
            model.train()
            start= i*batchsize
            end  = ((i+1)*batchsize if (i+1)*batchsize<datasize else datasize)
            y=TrainY[start:end]
            outputs = model(Train_c[start:end], Train_p[start:end], Train_t[start:end],-1)
            #od_loss=grad(outputs,Train_c,optimizer,model,start)
            optimizer.zero_grad()
            loss = loss_fn(outputs, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            batch+=1
            print("\r","Epoch: %d || batch:%d train_loss: %.10f"
            % (epoch+1, batch, loss.item()), end='', flush=True)

        # valid after each training epoch
        model.eval()
        Validation_output=model(Validation_c,Validation_p,Validation_t,-1)
        val_loss = loss_fn(Validation_output, ValidationY)
        if last_loss>val_loss:
            last_loss=val_loss
            #torch.save(model.state_dict(), './result/best_model.pkl')
            torch.save(model, './result/best_model.pkl')
        print('\n'+'*'*40)
        print("Epoch: %d || validation_loss: %.10f"
              % (epoch + 1, val_loss.item()))
        print('*' * 40)

    # *******************#
   #  # test
   #  # load model
   # # model = ST_resnet(c_dim, p_dim, t_dim, residual_units=2, day_dim=33)
   #  #load test data
   #  Test_c, Test_p, Test_t= (torch.from_numpy(TestX[0])
   #  , torch.from_numpy(TestX[1]), torch.from_numpy(TestX[2]))
   #  TestY = torch.from_numpy(TestY)
   #  # to cuda
   #  Test_c = Test_c.type(torch.FloatTensor).to('cuda')
   #  Test_p = Test_p.type(torch.FloatTensor).to('cuda')
   #  Test_t = Test_t.type(torch.FloatTensor).to('cuda')
   #  TestY = TestY.type(torch.FloatTensor).to('cuda')
   #  #load model
   #  model.load_state_dict(torch.load('./result/best_model.pkl'))
   #  model.eval()
   #  loss_fn=nn.MSELoss()
   #  batchsize= 50
   #  datasize= Test_c.shape[0]
   #  len =int(datasize/batchsize)+1
   #  index = torch.randperm(len)
   #  mse_loss=0
   #  n=0
   #  for i  in index:
   #      start= i*batchsize
   #      end  = ((i+1)*batchsize if (i+1)*batchsize<datasize else datasize)
   #      y=TestY[start:end]
   #      outputs = model(Test_c[start:end], Test_p[start:end], Test_t[start:end],-1)
   #      loss = loss_fn(outputs, y)
   #      mse_loss+=loss.item()
   #  print("Test_loss: %.10f" %(mse_loss/len))
   #  print("Test_Rescaled_loss: %.10f" % (mse_loss/len*MAX_FLOWIO**2))
   #  # write result
   #  f = open('./result/prediction_scores.txt', 'a')
   #  f.seek(0)
   #  f.truncate()  # 清空文件
   #  f.write("Keras MSE on test, %f\n" % (mse_loss/len))
   #  f.write("Rescaled MSE on test, %f\n" %(mse_loss/len*MAX_FLOWIO**2))
   #  f.close()
