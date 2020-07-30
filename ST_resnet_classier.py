import  numpy  as np
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from torchsummary import summary
from    collections import OrderedDict
from  torch.autograd import Function
from kmeans_pytorch import kmeans
from  sklearn.cluster import  KMeans
from params import *
class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias= True)
class _residual_unit(nn.Module):
    def __init__(self, nb_filter):
        super(_residual_unit, self).__init__()
        self.bn1=nn.BatchNorm2d(nb_filter,affine=True)
        self.conv=conv3x3(nb_filter,nb_filter)
        self.relu=torch.relu

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv(x)
        out += residual # short cut
        return out

class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x

class ST_resnet(nn.Module):
    '''
        Input
        C - Temperal closeness
        P - Period
        T - Trend
        dim = (map_height, map_width, nb_channel, len_sequence)
        number of residual units

        Return
        ST-ResNet model
    '''
    #def __init__(self,c_dim=(6,10,20),p_dim=(2,10,20),t_dim=(2,10,20),residual_units=2,day_dim=57):
    def __init__(self, c_dim, p_dim, t_dim, residual_units, day_dim):
        super(ST_resnet,self).__init__()
        self.residual_units=residual_units
        self.day_dim=day_dim
        self.relu = torch.relu
        self.tanh = torch.tanh
        self.c_dim=c_dim
        self.num_clusters=n_clusters
        self.linear1=nn.Linear(self.num_clusters,1,bias=False)
        self.linear2=nn.Linear(self.num_clusters,1,bias=False)
        self.conv1=nn.Conv2d(2, 32, kernel_size=3,stride=1, padding=1, bias= True)
        self.conv2=nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=True)
        _,self.map_height,self.map_width=c_dim
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        if day_dim>0:
            self.day_info = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.day_dim, 10, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, 2 * self.map_height * self.map_width, bias=True)),
            ('relu2', nn.ReLU()),
        ]))
        self.inputs_c = forward_network(self.c_dim[0],self.c_dim[0],self.c_dim[1],self.c_dim[2],residual_units)
        self.inputs_p = forward_network(p_dim[0], p_dim[0], p_dim[1], p_dim[2], residual_units)
        self.inputs_t = forward_network(t_dim[0], t_dim[0], t_dim[1], t_dim[2], residual_units)
    def forward(self,input_c, input_p, input_t, input_dayinfo):
        output=0
        input_c.require_grad=False
        batch=input_c.shape[0]
        len=input_c.shape[1]
        c=torch.zeros([batch,2,19,18]).cuda()
        inflow,outflow=[],[]
        for i in range(batch):
            inflow.append(input_c[i,0:int(len)-1:2].view(1,int(len/2),19*18).permute(0,2,1))
            outflow.append(input_c[i,1:int(len):2].view(1,int(len/2),19*18).permute(0,2,1))
        #cat
        inflow=torch.cat(inflow,dim=0).view(batch*19*18,int(len/2))
        outflow=torch.cat(outflow, dim=0).view(batch*19*18,int(len/2))
        z=self.linear1(one_hot_inflow).view(batch,19,18)
        c[:,0]=input_c[:,-1]*(1+z)
        z=self.linear2(one_hot_outflow).view(batch,19,18)
        c[:,1]=input_c[:,-2]*(1+z)
        output += self.inputs_c(c)
        #output += self.inputs_p(input_p)
        #output += self.inputs_t(input_t)
        if self.day_dim>0:
            day_output=self.day_info(input_dayinfo)
            day_output=day_output.view(-1, 2, self.map_height, self.map_width)
            output+=day_output

        output = self.relu(output)
        return output



# Matrix-based fusion
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1,2, h, w),
                                    requires_grad = True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights # element-wise multiplication
        return x

def forward_network(in_channels,nb_flow,map_height,map_width,nb_residual_unit):
    out_channels= 64
    return nn.Sequential(OrderedDict([
        ('conv',nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, padding=1, bias= True)),
        ('ResUnits', ResUnits(_residual_unit, nb_filter = 64, repetations = nb_residual_unit)),
        ('relu', nn.ReLU()),
        ('conv1', nn.Conv2d(out_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)),
        ('conv2', nn.Conv2d(out_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)),
        #('conv1', conv3x3(out_channels = 64, out_channels = 2)),
        #('FusionLayer', TrainableEltwiseLayer(n = 2, h = map_height, w = map_width))
    ]))
from  sklearn.preprocessing import LabelEncoder
from  sklearn.preprocessing import OneHotEncoder
def one_hot_line(flow):
    flow=torch.cat(flow,dim=0)
    batch=flow.shape[0]
    line=(flow[:,:,1:]-flow[:,:,-1:])
    line[line>0]=1
    line[line<0]=0
    #encoder
    onehot_encoder=torch.zeros(batch,flow.shape[1],2**(flow.shape[2]-1)).cuda()
    for i in range(batch):
        y=(line[i, :, 0] * 1 + line[i, :, 1] * 2 + line[i, :, 2] * 4 + line[i, :, 3] * 8).type(torch.long).cuda()
        onehot_encoder[i*batch:(i+1)*batch]=torch.eye(2**(flow.shape[2]-1))[y.reshape(-1)]
    return onehot_encoder


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    model = ST_resnet((6,19,18),(2,19,18),(2,19,18),1,57).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params:', pytorch_total_params)
    #print(model)
    summary(model,[(6,19,18),(2,19,18),(2,19,18),(1,1,57)])