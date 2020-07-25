import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models





def show_cam_on_image(img, mask,file):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(file, np.uint8(255 * cam))


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)
def process_grad(grad,img):
    grad=np.abs(grad)
    grad = np.maximum(grad, 0)
    grad = cv2.resize(grad, img.shape[:-1])
    grad = grad - np.min(grad)
    grad = grad / np.max(grad)
    grad = np.flip(grad,0)
    return grad

if __name__ == '__main__':

    img = cv2.imread('../map.png', 1)
    img = np.float32(cv2.resize(img, (785, 785))) / 255
    # #process grad
    grad=np.load('Test_c_grad.npy')[0,5,3,0]
    save_file='./plot_result/grad.jpg'
    grad = process_grad(grad,img)
    show_cam_on_image(img, grad,save_file)
    #guide bp
    guide_bp=np.load('../guide_BP_result/Test_c_grad.npy')[0,1,3,0]
    guide_bp = process_grad(guide_bp, img)
    save_file = './plot_result/guide_bp.jpg'
    show_cam_on_image(img, guide_bp,save_file)
    gb=cv2.merge([guide_bp,guide_bp,guide_bp])
    cam_mask = cv2.merge([grad, grad, grad])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)
    #ablation
    add_grad_loss=np.load('add_grad_loss.npy')
    origin_loss=np.load('origin_loss.npy')
    loss=(add_grad_loss-origin_loss)*52
    loss = process_grad(loss, img)
    save_file = './plot_result/add_loss.jpg'
    show_cam_on_image(img, loss,save_file)
    # intergrade
    intergrade_guide_bp=np.load('../guide_BP_result/Intergrated_grad_guide_bp.npy')[0]
    loss = process_grad(intergrade_guide_bp, img)
    save_file = './plot_result/Intergrated_grad_guide_bp.jpg'
    show_cam_on_image(img, loss,save_file)
    # intergrade
    intergrade_guide_bp=np.load('../result/Intergrated_grad.npy')[0]
    loss = process_grad(intergrade_guide_bp, img)
    save_file = './plot_result/Intergrated_grad.jpg'
    show_cam_on_image(img, loss,save_file)
    #grouthtrue
    grouthtrue=np.load('../data/od_10min_8.npy')[48]
    loss = process_grad(grouthtrue, img)
    save_file = './plot_result/grouthtrue.jpg'
    show_cam_on_image(img, loss,save_file)
    #input
    inputs=np.load('../guide_BP_result/input_Test_c.npy')[2]
    loss = process_grad(inputs, img)
    save_file = './plot_result/inputs.jpg'
    show_cam_on_image(img, loss,save_file)