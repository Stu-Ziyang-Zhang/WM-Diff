import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy

def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg

def concat_result(ori_img, pred_res, gt):
    if len(ori_img.shape) == 2:
        ori_img = np.expand_dims(ori_img, axis=0)
    if len(pred_res.shape) == 2:
        pred_res = np.expand_dims(pred_res, axis=0)
    if len(gt.shape) == 2:
        gt = np.expand_dims(gt, axis=0)
        
    assert len(ori_img.shape) == 3, f"Input image should be 3D, got: {ori_img.shape}"
    assert len(pred_res.shape) == 3, f"Prediction should be 3D, got: {pred_res.shape}"
    assert len(gt.shape) == 3, f"Ground truth should be 3D, got: {gt.shape}"
    ori_img = np.transpose(ori_img, (1,2,0))
    pred_res = np.transpose(pred_res, (1,2,0))
    gt = np.transpose(gt, (1,2,0))

    binary = deepcopy(pred_res)
    binary[binary >= 0.5] = 1
    binary[binary < 0.5] = 0

    if ori_img.shape[2] == 3:
        pred_res = np.repeat((pred_res * 255).astype(np.uint8), repeats=3, axis=2)
        binary = np.repeat((binary * 255).astype(np.uint8), repeats=3, axis=2)
        gt = np.repeat((gt * 255).astype(np.uint8), repeats=3, axis=2)
    total_img = np.concatenate((ori_img, pred_res, binary, gt), axis=1)
    return total_img

def save_img(data, filename):
    import os
    
    assert len(data.shape) == 3, "Image should be 3D (height*width*channels)"
    if data.shape[2] == 1:
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    img = Image.fromarray(data.astype(np.uint8))
    img.save(filename)
    return img