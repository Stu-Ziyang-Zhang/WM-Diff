import random
import os
import tempfile
from os.path import join
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from lib.extract_patches import get_data_train
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset
from lib.metrics import Evaluate
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2

def get_dataloader(args):
    patches_imgs_train, patches_masks_train = get_data_train(
        data_path_list = args.train_data_path_list,
        patch_height = args.train_patch_height,
        patch_width = args.train_patch_width,
        N_patches = args.N_patches,
        inside_FOV = args.inside_FOV
    )
    val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(args.val_ratio*patches_masks_train.shape[0])))
    train_ind =  set(range(patches_masks_train.shape[0])) - set(val_ind)
    train_ind = list(train_ind)

    train_set = TrainDataset(patches_imgs_train[train_ind,...],patches_masks_train[train_ind,...],mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=6)

    val_set = TrainDataset(patches_imgs_train[val_ind,...],patches_masks_train[val_ind,...],mode="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=6)
    if args.sample_visualization:
        N_sample = min(patches_imgs_train.shape[0], 50)
        sample_dir = join(args.outf, args.save)
        os.makedirs(sample_dir, exist_ok=True)
        
        save_img(group_images((patches_imgs_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(sample_dir, "sample_input_imgs.png"))
        save_img(group_images((patches_masks_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(sample_dir, "sample_input_masks.png"))
    return train_loader,val_loader


def get_dataloaderV2(args):
    """Load images to memory and create patch indices (lower memory usage)"""
    train_data_path = args.train_data_path_list
    temp_path = None
    
    if isinstance(train_data_path, list):
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        for path in train_data_path:
            if isinstance(path, str):
                temp_file.write(f"{path}\n")
        temp_file.close()
        temp_path = temp_file.name
        train_data_path = temp_path
    
    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list=train_data_path)

    patches_idx = create_patch_idx(fovs_train, args)

    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))

    train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="train",args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    if args.sample_visualization:
        visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
        visual_loader = DataLoader(visual_set, batch_size=1, shuffle=True, num_workers=0)
        N_sample = 50
        visual_imgs = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))

        for i, (img, mask) in enumerate(visual_loader):
            visual_imgs[i] = np.squeeze(img.numpy(), axis=0)
            visual_masks[i, 0] = np.squeeze(mask.numpy(), axis=0)
            if i >= N_sample - 1:
                break
            
        sample_dir = join(args.outf, args.save)
        os.makedirs(sample_dir, exist_ok=True)
        
        save_img(group_images((visual_imgs[:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(sample_dir, "sample_input_imgs.png"))
        save_img(group_images((visual_masks[:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(sample_dir, "sample_input_masks.png"))
    
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except Exception:
            pass
            
    return train_loader,val_loader

def train(train_loader, net, criterion, optimizer, device):
    """Training function"""
    import torch
    net.train()
    train_loss = AverageMeter()

    for inputs, targets in tqdm(train_loader, total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        has_nan_grad = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                          for p in net.parameters() if p.grad is not None)
        if has_nan_grad:
            optimizer.zero_grad()
            continue
                
        optimizer.step()
        train_loss.update(loss.item(), inputs.size(0))
    
    return OrderedDict([('train_loss', train_loss.avg)])

def val(val_loader, net, criterion, device):
    """Validation function"""
    import torch
    net.eval()
    val_loss = AverageMeter()
    evaluator = Evaluate()
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs_np = outputs.data.cpu().numpy()
            targets_np = targets.data.cpu().numpy()
            evaluator.add_batch(targets_np, outputs_np[:, 1])
    
    _, accuracy, _, _, _ = evaluator.confusion_matrix()
    return OrderedDict([
        ('val_loss', val_loss.avg),
        ('val_acc', accuracy),
        ('val_f1', evaluator.f1_score()),
        ('val_auc_roc', evaluator.auc_roc())
    ])