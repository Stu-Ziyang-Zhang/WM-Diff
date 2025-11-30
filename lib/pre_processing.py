import numpy as np
import cv2

def my_PreProc(data):
    assert len(data.shape) == 4
    assert data.shape[1] == 3, "Input should be RGB images"
    
    train_imgs = rgb2gray(data)
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.
    return train_imgs

def rgb2gray(rgb):
    assert len(rgb.shape) == 4, "Input should be 4D array"
    assert rgb.shape[1] == 3, "Input should have 3 channels (RGB)"
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs

def histo_equalized(imgs):
    assert len(imgs.shape) == 4, "Input should be 4D array"
    assert imgs.shape[1] == 1, "Input should have 1 channel"
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def clahe_equalized(imgs):
    assert len(imgs.shape) == 4
    assert imgs.shape[1] == 1
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def dataset_normalized(imgs):
    assert len(imgs.shape) == 4, "Input should be 4D array"
    assert imgs.shape[1] == 1, "Input should have 1 channel"
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert len(imgs.shape) == 4
    assert imgs.shape[1] == 1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

