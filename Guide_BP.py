from torch.autograd import Function
import torch
import  numpy as np
import torch.nn as nn
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


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, c,p,t,dayinfo):
        return self.model(c,p,t,dayinfo)

    def __call__(self, c,p,t,dayinfo):
        output = self.forward(c,p,t,dayinfo)

        return output


od_label=np.load('./data/all_od_7days.npy')
od_label=torch.from_numpy(od_label)
od_label= od_label.type(torch.FloatTensor).to('cuda')
nn_lose_=nn.CrossEntropyLoss()
softmax=nn.Softmax(dim=1)

def grad(outputs,val_c,optimizer,model,start):
    optimizer.zero_grad()
    end=start+outputs.shape[0]
    out=torch.sum(outputs[:,0,5,3])
    out.backward(retain_graph=True)
    grad=torch.sum(val_c.grad[start:end,1:2:5],1)
    #print(torch.max(grad))
    grad=torch.max(grad, torch.ones_like(grad) - 1)
    grad=grad.flatten(start_dim=1)
    grad=softmax(grad)
    label=softmax(od_label[start:end])
    loss=torch.sum(label*torch.log(label/grad))
    return loss

