from params import *
import  numpy as np
from process import *
import torch
from Guide_BP import *
from PIL import Image, ImageFilter
import cv2
import os
# data dim
c_dim = (nb_channel* len_c,HEIGHT, WIDTH)
p_dim = (nb_channel*len_p,HEIGHT, WIDTH)
t_dim = (nb_channel*len_t,HEIGHT, WIDTH)

# load data
TrainX, TrainY, TestX, TestY = process_data()
Test_c, Test_p, Test_t= (torch.from_numpy(TrainX[0])
                          , torch.from_numpy(TrainX[1]), torch.from_numpy(TrainX[2]))
#load model
model=torch.load('./result/best_model.pkl').inputs_c.train()
def vision_cnn(select_layer,resnet_layer,test_c,optimizer,model):
    maps=cv2.imread('map.png',1)
    maps=np.float32(cv2.resize(maps,(512,512)))/255
    x=test_c
    floder1='./image/cnn_layer_'+str(select_layer)
    floder2='./gray_images/cnn_layer_' + str(select_layer)
    for index, layer in enumerate(model):
        if index==1 or index==2:
            for indexs, layers in enumerate(layer.stacked_resunits[0]):
                x = layers(x)
                if resnet_layer==indexs and index==select_layer:
                    floder1='./image/resnet_'+str(select_layer)+'_cnn_layer_'+str(resnet_layer)
                    floder2='./gray_images/resnet_' + str(select_layer) + '_cnn_layer_' + str(resnet_layer)
                    break
        else:
            x = layer(x)
        if index==select_layer:
            break
    folders = os.path.exists(floder1)
    if not folders:
        os.makedirs(floder1)
    folders = os.path.exists(floder2)
    if not folders:
        os.makedirs(floder2)
    imag=np.zeros([x.shape[1],2,19,18])
    for i in range(x.shape[1]):
        optimizer.zero_grad()
        a=x[0,i].flatten()
        index=torch.argmax(a)
        a[index].backward(retain_graph=True)
        a[index]=-1e-10
        imag[i]=recreate_image(test_c.grad)
        #second max activation
        optimizer.zero_grad()
        index=torch.argmax(a)
        a[index].backward(retain_graph=True)
        imag[i]+=recreate_image(test_c.grad)
        for j in range(2):
            imags = np.flip(np.repeat(imag[i,j][np.newaxis], 3, axis=0).astype(np.uint8).transpose(1,2,0),axis=0)
            imags=cv2.resize(imags,(512,512))
            #im = Image.fromarray(imags)
            heatmap=cv2.applyColorMap(imags,2)
            heatmap=np.float32(heatmap)/255
            cam=cv2.add(0.5*heatmap,0.5*maps)
            path=floder1+'/conv_'+str(i)+'_channels_'+str(j)+'.jpg'
            cv2.imwrite(path,np.uint8(255*cam))
            path=floder2 + '/conv_' + str(i) + '_channels_' + str(j) + '.jpg'
            cv2.imwrite(path,imags)
    return imag
def recreate_image(im_as_var):
    recreated_im = im_as_var.cpu().numpy()[0]
    reverse_mean = np.mean(recreated_im,axis=(1,2))
    reverse_std =  np.std(recreated_im,axis=(1,2))
    for c in range(2):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im)
    return recreated_im

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)
timeindex=6*9
if __name__=='__main__':
    l=[0,1,1,2,2,4,5]
    s=[0,2,5,2,5,0,0]
    z=list(zip(l,s))
    test_c = nn.Parameter(Test_c[timeindex].unsqueeze(0).type(torch.FloatTensor).cuda())
    optimizer = torch.optim.Adam(list(model.parameters()) + [test_c])
    guide_model=GuidedBackpropReLUModel(model)
    for x,y in z:
        res=vision_cnn(select_layer=x,resnet_layer=y,test_c=test_c,optimizer=optimizer,model=guide_model.model)





