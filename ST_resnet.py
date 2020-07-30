import  numpy  as np
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from torchsummary import summary
from    collections import OrderedDict
from  torch.autograd import Function
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
def sequential_residual_unit(nb_filter):
    return nn.Sequential(OrderedDict([
        ('bn1',nn.BatchNorm2d(nb_filter,affine=True)),
        ('relu1',nn.ReLU()),
        ('conv1',conv3x3(nb_filter,nb_filter)),
        ('bn2', nn.BatchNorm2d(nb_filter, affine=True)),
        ('relu2', nn.ReLU()),
        ('conv2', conv3x3(nb_filter, nb_filter)),
    ]))



class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter)
    def make_stack_resunits(self, residual_unit, nb_filter):
        layers = []
        layers.append(residual_unit(nb_filter))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)+x
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
        self.rnn1 = nn.RNN(1,10,2,bias=False)
        self.rnn2 = nn.RNN(1,10,2,bias=False)
        self.linear1=nn.Linear(10,1,bias=False)
        self.linear2=nn.Linear(10,1,bias=False)
        self.conv1=nn.Conv2d(2, 32, kernel_size=3,stride=1, padding=1, bias= True)
        self.conv2=nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=True)
        _,self.map_height,self.map_width=c_dim
        if day_dim>0:
            self.day_info = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.day_dim, 10, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, 2 * self.map_height * self.map_width, bias=True)),
            ('relu2', nn.ReLU()),
        ]))
        self.inputs_c = forward_network(c_dim[0], c_dim[0], c_dim[1], c_dim[2], residual_units)
        self.inputs_p = forward_network(p_dim[0], p_dim[0], p_dim[1], p_dim[2], residual_units)
        self.inputs_t = forward_network(t_dim[0], t_dim[0], t_dim[1], t_dim[2], residual_units)
    def forward(self,input_c, input_p, input_t, input_dayinfo):
        output=0
        output += self.inputs_c(input_c)
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
    out_channels= 16
    return nn.Sequential(OrderedDict([
        ('conv',nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, padding=1, bias= True)),
        ('ResUnits1', ResUnits(sequential_residual_unit, nb_filter = out_channels)),
        ('ResUnits2', ResUnits(sequential_residual_unit, nb_filter = out_channels)),
        ('relu', nn.ReLU()),
        ('conv1', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)),
        ('conv2', nn.Conv2d(out_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)),
        #('conv1', conv3x3(out_channels = 64, out_channels = 2)),
        #('FusionLayer', TrainableEltwiseLayer(n = 2, h = map_height, w = map_width))
    ]))

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    model = ST_resnet((6,19,18),(2,19,18),(2,19,18),2,57).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params:', pytorch_total_params)
    #print(model)
    summary(model,[(6,19,18),(2,19,18),(2,19,18),(1,1,57)])