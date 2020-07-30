import matplotlib.pyplot as plt
from gradients import *
import numpy as np
import cv2
def show_cam_on_image(img, mask,file):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    #cv2.imwrite(file, np.uint8(255 * cam))
    return np.uint8(255 * cam)

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)
def process_grad(grad,img):
    #grad=np.abs(grad)
    #grad = np.maximum(grad, 0)
    grad = cv2.resize(grad, img.shape[:-1])
    grad = grad - np.min(grad)
    grad = grad / np.max(grad)
    grad = np.flip(grad,0)
    return grad

x=10
y=10
time=48
nums_step=10

if __name__=='__main__':
    img = cv2.imread('map.png', 1)
    img = np.float32(cv2.resize(img, (785, 785))) / 255
    grad,ablation_out= gradient(x, y, time, nums_step)
    ground_true=np.load('./data/all_od_10min_8.npy')[time]
    _, ax = plt.subplots(2, 6, figsize=(15, 8))
    save_file = ''
    title=['grad_in','grad_out','gb_in','gb_out','intergrad_in','intergrad_out','intergrad_gb_in','intergrad_out']
    for i in range(4):
        inflow=process_grad(grad[i*2,0],img)
        outflow=process_grad(grad[i*2+1,0],img)
        image1=show_cam_on_image(img, inflow, save_file)
        image2=show_cam_on_image(img, outflow, save_file)
        ax[0,i].imshow(image1)
        ax[0,i].set_title(title[i*2])
        plt.axis('off')
        ax[1,i].imshow(image2)
        ax[1,i].set_title(title[i*2+1])
        plt.axis('off')
    #ablation
    inflow =  process_grad(ablation_out[0],img)
    outflow = process_grad(ablation_out[1], img)
    image1 = show_cam_on_image(img, inflow, save_file)
    image2 = show_cam_on_image(img, outflow, save_file)
    ax[0, 4].imshow(image1)
    ax[0, 4].set_title('ablation_in')
    ax[1, 4].imshow(image2)
    ax[1, 4].set_title('ablation_out')
    #ground true
    inflow =  process_grad(ground_true[0],img)
    outflow = process_grad(ground_true[1], img)
    image1 = show_cam_on_image(img, inflow, save_file)
    image2 = show_cam_on_image(img, outflow, save_file)
    ax[0, 5].imshow(image1)
    ax[0, 5].set_title('ground_true_in')
    plt.axis('off')
    ax[1, 5].imshow(image2)
    ax[1, 5].set_title('ground_true_out')
    plt.axis('off')
    plt.show()
    _, ax = plt.subplots(2, 6, figsize=(15, 8))
    for i in range(4):
        inflow=process_grad(grad[i*2,1],img)
        outflow=process_grad(grad[i*2+1,1],img)
        image1=show_cam_on_image(img, inflow, save_file)
        image2=show_cam_on_image(img, outflow, save_file)
        ax[0,i].imshow(image1)
        ax[0,i].set_title(title[i*2])
        plt.axis('off')
        ax[1,i].imshow(image2)
        ax[1,i].set_title(title[i*2+1])
        plt.axis('off')
    #ablation
    inflow =  process_grad(ablation_out[0],img)
    outflow = process_grad(ablation_out[1], img)
    image1 = show_cam_on_image(img, inflow, save_file)
    image2 = show_cam_on_image(img, outflow, save_file)
    ax[0, 4].imshow(image1)
    ax[0, 4].set_title('ablation_in')
    ax[1, 4].imshow(image2)
    ax[1, 4].set_title('ablation_out')
    #ground true
    inflow =  process_grad(ground_true[0],img)
    outflow = process_grad(ground_true[1], img)
    image1 = show_cam_on_image(img, inflow, save_file)
    image2 = show_cam_on_image(img, outflow, save_file)
    ax[0, 5].imshow(image1)
    ax[0, 5].set_title('ground_true_in')
    plt.axis('off')
    ax[1, 5].imshow(image2)
    ax[1, 5].set_title('ground_true_out')
    plt.axis('off')
    plt.show()

