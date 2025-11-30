import numpy as np
import random
import configparser
import matplotlib.pyplot as plt
import os

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from .visualize import save_img, group_images
from .common import readImg
from .pre_processing import my_PreProc

def load_file_path_txt(file_path):
    img_list = []
    gt_list = []
    fov_list = []
    
    if isinstance(file_path, list):
        for line in file_path:
            items = line.strip().split(' ')
            if len(items) >= 3:
                img, gt, fov = items[0], items[1], items[2]
                img_list.append(img)
                gt_list.append(gt)
                fov_list.append(fov)
    else:
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                img, gt, fov = lines.split(' ')
                img_list.append(img)
                gt_list.append(gt)
                fov_list.append(fov)
    
    return img_list, gt_list, fov_list

def load_data(data_path_list_file):
    img_list, gt_list, fov_list = load_file_path_txt(data_path_list_file)
    imgs = None
    groundTruth = None
    FOVs = None
    for i in range(len(img_list)):
        img = np.asarray(readImg(img_list[i]))
        gt = np.asarray(readImg(gt_list[i]))
        if len(gt.shape)==3:
            gt = gt[:,:,0]
        fov = np.asarray(readImg(fov_list[i]))
        if len(fov.shape)==3:
            fov = fov[:,:,0]

        imgs = np.expand_dims(img, 0) if imgs is None else np.concatenate((imgs, np.expand_dims(img, 0)))
        groundTruth = np.expand_dims(gt, 0) if groundTruth is None else np.concatenate((groundTruth, np.expand_dims(gt, 0)))
        FOVs = np.expand_dims(fov, 0) if FOVs is None else np.concatenate((FOVs, np.expand_dims(fov, 0)))

    if np.max(FOVs) <= 1:
        FOVs = FOVs * 255
    
    assert(np.min(FOVs) == 0 and (np.max(FOVs) == 255 or np.max(FOVs) == 1))
    
    assert((np.min(groundTruth) == 0 and (np.max(groundTruth) == 255 or np.max(groundTruth) == 1)))
    if np.max(groundTruth) == 1:
        groundTruth = groundTruth * 255

    imgs = np.transpose(imgs, (0, 3, 1, 2))
    groundTruth = np.expand_dims(groundTruth, 1)
    FOVs = np.expand_dims(FOVs, 1)
    return imgs, groundTruth, FOVs

def get_data_train(data_path_list, patch_height, patch_width, N_patches, inside_FOV):
    train_imgs_original, train_masks, train_FOVs = load_data(data_path_list)
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.
    train_FOVs = train_FOVs // 255

    data_dim_check(train_imgs, train_masks)
    assert(np.min(train_masks) == 0 and np.max(train_masks) == 1)
    assert(np.min(train_FOVs) == 0 and np.max(train_FOVs) == 1)

    patches_imgs_train, patches_masks_train = extract_random(train_imgs, train_masks, train_FOVs, patch_height, patch_width, N_patches, inside_FOV)
    data_dim_check(patches_imgs_train, patches_masks_train)

    return patches_imgs_train, patches_masks_train

def extract_random(full_imgs, full_masks, full_FOVs, patch_h, patch_w, N_patches, inside='not'):
    patch_per_img = int(N_patches / full_imgs.shape[0])
    if (N_patches % full_imgs.shape[0] != 0):
        N_patches = patch_per_img * full_imgs.shape[0]
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w), dtype=np.uint8)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]

    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        k = 0
        while k < patch_per_img:
            x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
            y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
            if inside == 'center' or inside == 'all':
                if not is_patch_inside_FOV(x_center, y_center, full_FOVs[i, 0], patch_h, patch_w, mode=inside):
                    continue
            patch = full_imgs[i, :, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i, :, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1
            k += 1
    return patches, patches_masks

def is_patch_inside_FOV(x, y, fov_img, patch_h, patch_w, mode='center'):
    if mode == 'center':
        return fov_img[y, x]
    elif mode == 'all':
        fov_patch = fov_img[y-int(patch_h/2):y+int(patch_h/2), x-int(patch_w/2):x+int(patch_w/2)]
        return fov_patch.all()
    else:
        return False

def data_dim_check(imgs, masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)

def get_data_test_overlap(test_data_path_list, patch_height, patch_width, stride_height, stride_width):
    test_imgs_original, test_masks, test_FOVs= load_data(test_data_path_list)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255. if np.max(test_masks) > 1 else test_masks
    test_FOVs = test_FOVs / 255. if np.max(test_FOVs) > 1 else test_FOVs
    
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    assert(np.max(test_masks) == 1 and np.min(test_masks) == 0)

    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    return patches_imgs_test, test_imgs_original, test_masks, test_FOVs, test_imgs.shape[2], test_imgs.shape[3]

def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    leftover_h = (img_h-patch_h)%stride_h
    leftover_w = (img_w-patch_w)%stride_w
    if (leftover_h != 0):
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    return full_imgs

def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1
    assert (iter_tot==N_patches_tot)
    return patches

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w, debug_dir=None, FOVs=None):
    import numpy as np
    import cv2
    from scipy.ndimage import gaussian_filter

    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    N_full_imgs = preds.shape[0] // N_patches_img

    is_torch = hasattr(preds, 'device')
    if is_torch:
        import torch
        device = preds.device
        final_result = torch.zeros((N_full_imgs, preds.shape[1], img_h, img_w), device=device)
    else:
        final_result = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0
    for i in range(N_full_imgs):
        if debug_dir is not None and i == 0:
            import os
            os.makedirs(debug_dir, exist_ok=True)
        if is_torch:
            import torch
            device = preds.device
            img_prob = torch.zeros((preds.shape[1], img_h, img_w), device=device)
            weight_map = torch.zeros((preds.shape[1], img_h, img_w), device=device)
            boundary_map = torch.zeros((preds.shape[1], img_h, img_w), device=device)
            
            multi_scale_prob = torch.zeros((preds.shape[1], img_h, img_w), device=device)
            multi_scale_weight = torch.zeros((preds.shape[1], img_h, img_w), device=device)
        else:
            img_prob = np.zeros((preds.shape[1], img_h, img_w))
            weight_map = np.zeros((preds.shape[1], img_h, img_w))
            boundary_map = np.zeros((preds.shape[1], img_h, img_w))
            multi_scale_prob = np.zeros((preds.shape[1], img_h, img_w))
            multi_scale_weight = np.zeros((preds.shape[1], img_h, img_w))

        weight_kernel = create_weight_kernel(patch_h, patch_w, is_torch, device if is_torch else None)
        patch_count = 0
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                h_start = h * stride_h
                h_end = h_start + patch_h
                w_start = w * stride_w
                w_end = w_start + patch_w

                if k < preds.shape[0]:
                    if is_torch:
                        weighted_pred = preds[k] * weight_kernel
                        img_prob[:, h_start:h_end, w_start:w_end] += weighted_pred
                        weight_map[:, h_start:h_end, w_start:w_end] += weight_kernel
                        
                        if preds[k].shape[1] > 2 and preds[k].shape[2] > 2:
                            import torch.nn.functional as F
                            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                 device=device).float().view(1, 1, 3, 3).repeat(preds.shape[1], 1, 1, 1)
                            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                 device=device).float().view(1, 1, 3, 3).repeat(preds.shape[1], 1, 1, 1)
                            patch_pad = F.pad(preds[k].unsqueeze(0), (1, 1, 1, 1), mode='replicate')
                            edge_x = F.conv2d(patch_pad, sobel_x, groups=preds.shape[1])
                            edge_y = F.conv2d(patch_pad, sobel_y, groups=preds.shape[1])
                            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0)
                            boundary_map[:, h_start:h_end, w_start:w_end] += edge_magnitude * weight_kernel
                    else:
                        weighted_pred = preds[k] * weight_kernel
                        img_prob[:, h_start:h_end, w_start:w_end] += weighted_pred
                        weight_map[:, h_start:h_end, w_start:w_end] += weight_kernel
                        
                        if preds[k].shape[1] > 2 and preds[k].shape[2] > 2:
                            for c in range(preds[k].shape[0]):
                                patch_normalized = (preds[k][c] * 255).astype(np.uint8)
                                edge_x = cv2.Sobel(patch_normalized, cv2.CV_32F, 1, 0, ksize=3)
                                edge_y = cv2.Sobel(patch_normalized, cv2.CV_32F, 0, 1, ksize=3)
                                edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
                                boundary_map[c, h_start:h_end, w_start:w_end] += edge_magnitude * weight_kernel[0]
                    
                    if patch_h >= 16 and patch_w >= 16:
                        down_factor = 2
                        down_h = patch_h // down_factor
                        down_w = patch_w // down_factor
                        
                        if is_torch:
                            import torch.nn.functional as F
                            down_pred = F.interpolate(preds[k].unsqueeze(0), 
                                                      size=(down_h, down_w), 
                                                      mode='bilinear', 
                                                      align_corners=False).squeeze(0)
                            down_weight = F.interpolate(weight_kernel.unsqueeze(0), 
                                                        size=(down_h, down_w), 
                                                        mode='bilinear', 
                                                        align_corners=False).squeeze(0)
                        else:
                            down_pred = np.zeros((preds.shape[1], down_h, down_w))
                            down_weight = np.zeros((1, down_h, down_w))
                            for c in range(preds.shape[1]):
                                down_pred[c] = cv2.resize(preds[k][c], (down_w, down_h), 
                                                          interpolation=cv2.INTER_LINEAR)
                            down_weight[0] = cv2.resize(weight_kernel[0], (down_w, down_h), 
                                                       interpolation=cv2.INTER_LINEAR)
                        
                        h_start_ms = h_start - (patch_h - down_h) // 2
                        h_end_ms = h_start_ms + patch_h
                        w_start_ms = w_start - (patch_w - down_w) // 2
                        w_end_ms = w_start_ms + patch_w
                        
                        if h_start_ms < 0: h_start_ms = 0
                        if w_start_ms < 0: w_start_ms = 0
                        if h_end_ms > img_h: h_end_ms = img_h
                        if w_end_ms > img_w: w_end_ms = img_w
                        
                        valid_h = h_end_ms - h_start_ms
                        valid_w = w_end_ms - w_start_ms
                        
                        if valid_h > 0 and valid_w > 0:
                            if is_torch:
                                if valid_h != patch_h or valid_w != patch_w:
                                    ms_pred = F.interpolate(down_pred.unsqueeze(0), 
                                                           size=(valid_h, valid_w), 
                                                           mode='bilinear', 
                                                           align_corners=False).squeeze(0)
                                    ms_weight = F.interpolate(down_weight.unsqueeze(0), 
                                                             size=(valid_h, valid_w), 
                                                             mode='bilinear', 
                                                             align_corners=False).squeeze(0)
                                else:
                                    ms_pred = F.interpolate(down_pred.unsqueeze(0), 
                                                           size=(patch_h, patch_w), 
                                                           mode='bilinear', 
                                                           align_corners=False).squeeze(0)
                                    ms_weight = F.interpolate(down_weight.unsqueeze(0), 
                                                             size=(patch_h, patch_w), 
                                                             mode='bilinear', 
                                                             align_corners=False).squeeze(0)
                                
                                multi_scale_prob[:, h_start_ms:h_end_ms, w_start_ms:w_end_ms] += ms_pred * ms_weight
                                multi_scale_weight[:, h_start_ms:h_end_ms, w_start_ms:w_end_ms] += ms_weight
                            else:
                                ms_pred = np.zeros((preds.shape[1], valid_h, valid_w))
                                ms_weight = np.zeros((1, valid_h, valid_w))
                                
                                for c in range(preds.shape[1]):
                                    if valid_h != patch_h or valid_w != patch_w:
                                        ms_pred[c] = cv2.resize(down_pred[c], (valid_w, valid_h), 
                                                               interpolation=cv2.INTER_LINEAR)
                                    else:
                                        ms_pred[c] = cv2.resize(down_pred[c], (patch_w, patch_h), 
                                                               interpolation=cv2.INTER_LINEAR)
                                
                                if valid_h != patch_h or valid_w != patch_w:
                                    ms_weight[0] = cv2.resize(down_weight[0], (valid_w, valid_h), 
                                                            interpolation=cv2.INTER_LINEAR)
                                else:
                                    ms_weight[0] = cv2.resize(down_weight[0], (patch_w, patch_h), 
                                                            interpolation=cv2.INTER_LINEAR)
                                
                                for c in range(preds.shape[1]):
                                    multi_scale_prob[c, h_start_ms:h_end_ms, w_start_ms:w_end_ms] += ms_pred[c] * ms_weight[0]
                                    multi_scale_weight[c, h_start_ms:h_end_ms, w_start_ms:w_end_ms] += ms_weight[0]
                    
                    k += 1
                    patch_count += 1

        if k - patch_count >= 0:
            k = k - patch_count
            for h in range(N_patches_h-1, -1, -1):
                for w in range(N_patches_w-1, -1, -1):
                    h_start = h * stride_h
                    h_end = h_start + patch_h
                    w_start = w * stride_w
                    w_end = w_start + patch_w
                    
                    if k < preds.shape[0]:
                        is_border = (h == 0 or h == N_patches_h-1 or w == 0 or w == N_patches_w-1)
                        
                        if is_torch:
                            border_weight = weight_kernel.clone()
                            if is_border:
                                border_weight = border_weight * 1.5
                            weighted_pred = preds[k] * border_weight
                            img_prob[:, h_start:h_end, w_start:w_end] += weighted_pred
                            weight_map[:, h_start:h_end, w_start:w_end] += border_weight
                        else:
                            border_weight = weight_kernel.copy()
                            if is_border:
                                border_weight = border_weight * 1.5
                            weighted_pred = preds[k] * border_weight
                            img_prob[:, h_start:h_end, w_start:w_end] += weighted_pred
                            weight_map[:, h_start:h_end, w_start:w_end] += border_weight
                        
                        k += 1

        if debug_dir is not None and i == 0:
            import os
            import matplotlib.pyplot as plt
            os.makedirs(debug_dir, exist_ok=True)

            if is_torch:
                img_prob_np = img_prob.detach().cpu().numpy()
                weight_map_np = weight_map.detach().cpu().numpy()
                boundary_map_np = boundary_map.detach().cpu().numpy()
                multi_scale_prob_np = multi_scale_prob.detach().cpu().numpy()
                multi_scale_weight_np = multi_scale_weight.detach().cpu().numpy()
            else:
                img_prob_np = img_prob
                weight_map_np = weight_map
                boundary_map_np = boundary_map
                multi_scale_prob_np = multi_scale_prob
                multi_scale_weight_np = multi_scale_weight

            np.save(os.path.join(debug_dir, "full_prob.npy"), img_prob_np[0])
            np.save(os.path.join(debug_dir, "weight_map.npy"), weight_map_np[0])
            np.save(os.path.join(debug_dir, "boundary_map.npy"), boundary_map_np[0])
            np.save(os.path.join(debug_dir, "multi_scale_prob.npy"), multi_scale_prob_np[0])

            plt.figure(figsize=(15, 10))
            plt.subplot(221)
            plt.imshow(img_prob_np[0], cmap='hot')
            plt.colorbar()
            plt.title("Weighted Accumulated Probability")
            plt.subplot(222)
            plt.imshow(weight_map_np[0], cmap='viridis')
            plt.colorbar()
            plt.title("Weight Map")
            plt.subplot(223)
            plt.imshow(boundary_map_np[0], cmap='plasma')
            plt.colorbar()
            plt.title("Boundary Feature Map")
            plt.subplot(224)
            plt.imshow(multi_scale_prob_np[0], cmap='inferno')
            plt.colorbar()
            plt.title("Multi-scale Fusion Probability")
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "recompone_arrays.png"))
            plt.close()
            
            if is_torch:
                wk_np = weight_kernel.detach().cpu().numpy()
            else:
                wk_np = weight_kernel
                
            plt.figure(figsize=(8, 8))
            plt.imshow(wk_np[0], cmap='jet')
            plt.colorbar()
            plt.title("Weight Kernel")
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "weight_kernel.png"))
            plt.close()

        if is_torch:
            import torch
            import torch.nn.functional as F
            
            weight_map = torch.clamp(weight_map, min=1e-8)
            img_avg = img_prob / weight_map
            
            multi_scale_weight = torch.clamp(multi_scale_weight, min=1e-8)
            multi_scale_avg = multi_scale_prob / multi_scale_weight
            
            edge_kernel_size = 5
            edge_kernel = torch.ones((1, 1, edge_kernel_size, edge_kernel_size), device=device) / (edge_kernel_size**2)
            
            grid_edges = torch.zeros_like(weight_map)
            grid_stride_h = patch_h - stride_h
            grid_stride_w = patch_w - stride_w
            
            for h_grid in range(0, img_h, stride_h):
                if h_grid > 0 and h_grid < img_h:
                    h_start = max(0, h_grid - grid_stride_h//2)
                    h_end = min(img_h, h_grid + grid_stride_h//2)
                    grid_edges[:, h_start:h_end, :] = 0.8
            
            for w_grid in range(0, img_w, stride_w):
                if w_grid > 0 and w_grid < img_w:
                    w_start = max(0, w_grid - grid_stride_w//2)
                    w_end = min(img_w, w_grid + grid_stride_w//2)
                    grid_edges[:, :, w_start:w_end] = 0.8
            
            fusion_weight = torch.sigmoid(3.0 * grid_edges)
            if fusion_weight.shape != img_avg.shape:
                if fusion_weight.shape[0] == img_avg.shape[0]:
                    fusion_weight = fusion_weight.expand_as(img_avg)
                else:
                    fusion_weight = torch.zeros_like(img_avg)
                    for c in range(img_avg.shape[0]):
                        fusion_weight[c] = torch.sigmoid(3.0 * grid_edges[min(c, grid_edges.shape[0]-1)])

            fused_result = img_avg * (1 - fusion_weight) + multi_scale_avg * fusion_weight
            
            smoothed = F.conv2d(
                fused_result.unsqueeze(0), 
                edge_kernel, 
                padding=edge_kernel_size//2
            ).squeeze(0)
            
            edge_mask = (weight_map < weight_map.mean()*0.7).float()
            fused_result = fused_result * (1 - edge_mask) + smoothed * edge_mask
            
            boundary_map = F.conv2d(
                boundary_map.unsqueeze(0), 
                edge_kernel, 
                padding=edge_kernel_size//2
            ).squeeze(0)
            
            if torch.max(boundary_map) > 1e-8:
                boundary_map = boundary_map / torch.max(boundary_map)
                edge_enhancement = 1.0 + boundary_map * 0.3
                fused_result = fused_result * edge_enhancement
            
            small_kernel = 3
            small_kernel_tensor = torch.ones((1, 1, small_kernel, small_kernel), device=device) / (small_kernel**2)
            smoothed_small = F.conv2d(
                fused_result.unsqueeze(0), 
                small_kernel_tensor, 
                padding=small_kernel//2
            ).squeeze(0)
            
            structure_weight = torch.abs(fused_result - smoothed_small)
            structure_mask = (structure_weight < structure_weight.mean() * 1.5).float()
            
            final_result[i] = fused_result * (1 - structure_mask * 0.5) + smoothed_small * (structure_mask * 0.5)
            
        else:
            from scipy.ndimage import gaussian_filter, uniform_filter
            import cv2
            
            weight_map = np.maximum(weight_map, 1e-8)
            img_avg = img_prob / weight_map
            
            multi_scale_weight = np.maximum(multi_scale_weight, 1e-8)
            multi_scale_avg = multi_scale_prob / multi_scale_weight
            
            grid_edges = np.zeros_like(weight_map)
            grid_stride_h = patch_h - stride_h
            grid_stride_w = patch_w - stride_w
            
            for h_grid in range(0, img_h, stride_h):
                if h_grid > 0 and h_grid < img_h:
                    h_start = max(0, h_grid - grid_stride_h//2)
                    h_end = min(img_h, h_grid + grid_stride_h//2)
                    grid_edges[:, h_start:h_end, :] = 0.8
            
            for w_grid in range(0, img_w, stride_w):
                if w_grid > 0 and w_grid < img_w:
                    w_start = max(0, w_grid - grid_stride_w//2)
                    w_end = min(img_w, w_grid + grid_stride_w//2)
                    grid_edges[:, :, w_start:w_end] = 0.8
            
            fusion_weight = 1 / (1 + np.exp(-3.0 * grid_edges))

            if fusion_weight.shape != img_avg.shape:
                if fusion_weight.shape[0] == img_avg.shape[0]:
                    new_fusion_weight = np.zeros_like(img_avg)
                    for c in range(fusion_weight.shape[0]):
                        new_fusion_weight[c] = fusion_weight[c]
                    fusion_weight = new_fusion_weight
                else:
                    new_fusion_weight = np.zeros_like(img_avg)
                    for c in range(img_avg.shape[0]):
                        new_fusion_weight[c] = 1 / (1 + np.exp(-3.0 * grid_edges[min(c, grid_edges.shape[0]-1)]))
                    fusion_weight = new_fusion_weight

            fused_result = img_avg * (1 - fusion_weight) + multi_scale_avg * fusion_weight
            
            smoothed = np.zeros_like(fused_result)
            edge_kernel_size = 5
            for c in range(fused_result.shape[0]):
                smoothed[c] = uniform_filter(fused_result[c], size=edge_kernel_size)
            
            edge_mask = (weight_map < np.mean(weight_map)*0.7).astype(np.float32)
            fused_result = fused_result * (1 - edge_mask) + smoothed * edge_mask
            
            boundary_smoothed = np.zeros_like(boundary_map)
            for c in range(boundary_map.shape[0]):
                boundary_smoothed[c] = uniform_filter(boundary_map[c], size=edge_kernel_size)
            boundary_map = boundary_smoothed
            
            if np.max(boundary_map) > 1e-8:
                boundary_map = boundary_map / np.max(boundary_map)
                edge_enhancement = 1.0 + boundary_map * 0.3
                fused_result = fused_result * edge_enhancement
            
            small_kernel = 3
            smoothed_small = np.zeros_like(fused_result)
            for c in range(fused_result.shape[0]):
                smoothed_small[c] = uniform_filter(fused_result[c], size=small_kernel)
            
            structure_weight = np.abs(fused_result - smoothed_small)
            structure_mask = (structure_weight < np.mean(structure_weight) * 1.5).astype(np.float32)
            
            final_result[i] = fused_result * (1 - structure_mask * 0.5) + smoothed_small * (structure_mask * 0.5)
            
            for c in range(final_result[i].shape[0]):
                final_result[i, c] = cv2.bilateralFilter(final_result[i, c].astype(np.float32), 7, 0.1, 5)


    if debug_dir is not None:
        import os
        import matplotlib.pyplot as plt
        if is_torch:
            final_np = final_result[0].detach().cpu().numpy()
        else:
            final_np = final_result[0]
        np.save(os.path.join(debug_dir, "final_avg.npy"), final_np[0])

        if final_np.ndim == 3:
            img2d = final_np[0]
        else:
            img2d = final_np
        plt.figure(figsize=(10, 8))
        plt.imshow(img2d, cmap='jet')
        plt.colorbar()
        plt.title("Final Reconstructed Result")
        plt.savefig(os.path.join(debug_dir, "final_result.png"))
        plt.close()

        h_mid = img_h // 2
        w_mid = img_w // 2
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(img2d[h_mid, :])
        plt.title(f"Horizontal Mid Slice (h={h_mid})")
        plt.grid(True)
        plt.subplot(122)
        plt.plot(img2d[:, w_mid])
        plt.title(f"Vertical Mid Slice (w={w_mid})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, "mid_slices.png"))
        plt.close()

    if FOVs is not None:
        kill_border(final_result, FOVs)
    
    if debug_dir is not None:
        if is_torch:
            final_np_fov = final_result[0].detach().cpu().numpy()
        else:
            final_np_fov = final_result[0]
            
        if final_np_fov.ndim == 3:
            img2d_fov = final_np_fov[0]
        else:
            img2d_fov = final_np_fov
            
        plt.figure(figsize=(10, 8))
        plt.imshow(img2d_fov, cmap='jet')
        plt.colorbar()
        plt.title("Result with FOV Mask")
        plt.savefig(os.path.join(debug_dir, "final_result_with_fov.png"))
        plt.close()
    return final_result


def create_weight_kernel(patch_h, patch_w, is_torch=False, device=None):
    import numpy as np
    
    y = np.linspace(-3, 3, patch_h)
    x = np.linspace(-3, 3, patch_w)
    xx, yy = np.meshgrid(x, y)
    
    sigma = 2.0
    weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    r = np.sqrt(xx**2 + yy**2)
    r_max = np.max(r)
    radial_factor = 1.0 - 0.7 * (r / r_max)**2
    weight = weight * radial_factor
    
    boundary_width = max(5, min(patch_h, patch_w) // 10)
    edge_mask = np.ones_like(weight)
    for i in range(boundary_width):
        decay = (1 - (i / boundary_width)**2) * 0.7
        if i < patch_h:
            edge_mask[i, :] *= decay
            edge_mask[patch_h-i-1, :] *= decay
        if i < patch_w:
            edge_mask[:, i] *= decay
            edge_mask[:, patch_w-i-1] *= decay
    
    weight = weight * edge_mask
    weight = 0.01 + 0.99 * (weight - np.min(weight)) / (np.max(weight) - np.min(weight) + 1e-8)
    weight = weight.reshape(1, patch_h, patch_w)
    
    if is_torch:
        import torch
        weight = torch.tensor(weight, device=device, dtype=torch.float32)
    
    return weight

def pred_only_in_FOV(data_imgs, data_masks, FOVs):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):
        for x in range(width):
            for y in range(height):
                if pixel_inside_FOV(i,x,y,FOVs):
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

def kill_border(data, FOVs):
    assert (len(data.shape)==4)
    assert (data.shape[1]==1 or data.shape[1]==3)
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):
        for x in range(width):
            for y in range(height):
                if not pixel_inside_FOV(i,x,y,FOVs):
                    data[i,:,y,x]=0.0

def pixel_inside_FOV(i, x, y, FOVs):
    assert (len(FOVs.shape)==4)
    assert (FOVs.shape[1]==1)
    if (x >= FOVs.shape[3] or y >= FOVs.shape[2]):
        return False
    return FOVs[i,0,y,x]>0

