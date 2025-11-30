import os
import joblib
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
import PIL

def readImg(img_path):
    return PIL.Image.open(img_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_args(args, save_path):
    os.makedirs(save_path, exist_ok=True)

    print('Config info -----')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    args_file = os.path.join(save_path, 'args.txt')
    with open(args_file, 'w') as f:
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}', file=f)
    
    pkl_file = os.path.join(save_path, 'args.pkl')
    joblib.dump(args, pkl_file)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def dict_round(dic, num):
    for key, value in dic.items():
        dic[key] = round(value, num)
    return dic

