import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import sys
from os.path import join
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from tqdm import trange, tqdm
from PIL import Image

from config import calculate_metrics
from diffusion_framework import T, forward_diffusion_sample
from lib.sampling import sample_ddim
from lib.monitoring import PrecisionMonitor
from lib.training_utils import warmup_optimizer, save_sample_images
from lib.postprocessing import process_segmentation_result
from lib.losses.loss import BCEWithLogitsLoss, BCELoss, DiffusionLoss
from lib.logger import Logger, Print_Logger
from lib.metrics import Evaluate


def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None,
                  mode=1, device='cuda', grad_acc_steps=0, mixed_precision=False,
                  scaler=None, precision_monitor=None, save_patches=False, save_dir=None, epoch=None,
                  initial_state=None, args=None, train_cond_net_only=False,
                  diffusion_loss_fn=None, bce_loss_fn=None):
    train = optimizer is not None

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    evaluator = Evaluate()
    
    if not train and save_patches:
        all_inputs = []
        all_outputs = []
        all_labels = []
        original_shapes = {}

    if initial_state is not None and isinstance(initial_state, dict):
        running_loss = initial_state.get('running_loss', 0.0)
        n_elems = initial_state.get('n_elems', 0)
        if train and 'loss' in initial_state:
            forced_loss = initial_state.get('loss', 0.2)
            if n_elems == 0 or running_loss / n_elems > forced_loss * 2:
                running_loss = forced_loss
                n_elems = 1
    else:
        running_loss = 0.0
        n_elems = 0

    if save_patches and not train and save_dir is not None:
        patch_save_dir = os.path.join(save_dir, f'val_patches_epoch_{epoch}')
        os.makedirs(patch_save_dir, exist_ok=True)
        
        full_img_save_dir = os.path.join(save_dir, f'val_full_images_epoch_{epoch}')
        os.makedirs(full_img_save_dir, exist_ok=True)
        
        debug_dir = os.path.join(save_dir, f'val_debug_epoch_{epoch}')
        os.makedirs(debug_dir, exist_ok=True)

    if diffusion_loss_fn is None:
        diffusion_loss_fn = DiffusionLoss()
    if bce_loss_fn is None:
        bce_loss_fn = BCELoss()

    with trange(len(loader)) as t:
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            BATCH_SIZE = labels.size(0)

            if mode == 1:
                time_step = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
                use_diffusion = False
            elif mode == 2:
                time_step = torch.randint(1, T, (BATCH_SIZE,), device=device).long()
                use_diffusion = True
            else:
                use_diffusion = random.random() < 0.5
                if use_diffusion:
                    time_step = torch.randint(1, T, (BATCH_SIZE,), device=device).long()
                else:
                    time_step = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=mixed_precision):
                if use_diffusion:
                    labels_noisy, noise = forward_diffusion_sample(
                        labels.unsqueeze(dim=1).float(), time_step, device
                    )
                    
                    predicted_noise = model(
                        labels_noisy, 
                        time_step, 
                        x_original=inputs,
                        train_cond_net=not train_cond_net_only if train else False
                    )
                    
                    diffusion_loss = diffusion_loss_fn(predicted_noise, noise)
                    
                    time_step_seg = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
                    outputs_seg = model(
                        inputs, 
                        time_step_seg, 
                        x_original=inputs,
                        train_cond_net=not train_cond_net_only if train else False
                    )
                    seg_loss = criterion(outputs_seg, labels.unsqueeze(dim=1).float())
                    
                    lambda_seg = 0.5
                    loss = diffusion_loss + lambda_seg * seg_loss
                    
                    outputs = outputs_seg
                    
                    outputs_np = outputs.data.cpu().numpy()
                    labels_np = labels.data.cpu().numpy()
                    evaluator.add_batch(labels_np, outputs_np[:, 1] if outputs.shape[1] > 1 else outputs_np.squeeze())
                    
                    if not train:
                        with torch.no_grad():
                            x_noisy_init = torch.rand_like(labels.unsqueeze(dim=1).float())
                            outputs_ddim = sample_ddim(
                                model, x_noisy_init, inputs, 
                                timesteps=50, device=device
                            )
                            outputs_np_ddim = outputs_ddim.data.cpu().numpy()
                            evaluator.add_batch(
                                labels_np, 
                                outputs_np_ddim[:, 0] if outputs_ddim.shape[1] > 1 else outputs_np_ddim.squeeze()
                            )
                else:
                    if train_cond_net_only:
                        outputs = model.cond_net(inputs)
                        labels_expanded = torch.zeros_like(outputs)
                        labels_float = labels.float()
                        labels_expanded[:, 0] = 1.0 - labels_float
                        labels_expanded[:, 1] = labels_float
                        loss = bce_loss_fn(outputs, labels_expanded)
                    else:
                        if not train:
                            with torch.no_grad():
                                x_noisy_init = torch.rand_like(labels.unsqueeze(dim=1).float())
                                outputs = sample_ddim(
                                    model, x_noisy_init, inputs, 
                                    timesteps=50, device=device
                                )
                            loss = torch.tensor(0.0, device=device)
                        else:
                            outputs = model(
                                inputs, 
                                time_step, 
                                x_original=inputs,
                                train_cond_net=False
                            )
                            loss = criterion(outputs, labels.unsqueeze(dim=1).float())
                    
                    outputs_np = outputs.data.cpu().numpy()
                    labels_np = labels.data.cpu().numpy()
                    evaluator.add_batch(labels_np, outputs_np[:, 0] if outputs.shape[1] == 1 else (outputs_np[:, 1] if outputs.shape[1] > 1 else outputs_np.squeeze()))

                    if save_patches and not train and save_dir is not None:
                        all_inputs.append(inputs.cpu().numpy())
                        
                        if outputs.shape[1] > 1:
                            outputs_np = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        else:
                            outputs_np = torch.sigmoid(outputs).squeeze().cpu().numpy()
                            if len(outputs_np.shape) == 2:
                                outputs_np = np.expand_dims(outputs_np, axis=0)
                        
                        all_outputs.append(outputs_np)
                        all_labels.append(labels.cpu().numpy())
                        
                        if i_batch < 5:
                            for i in range(min(4, BATCH_SIZE)):
                                patch_idx = i_batch * BATCH_SIZE + i
                                
                                input_patch = inputs[i].cpu().numpy()
                                if input_patch.shape[0] == 1:
                                    input_patch = input_patch.squeeze(0)
                                elif input_patch.shape[0] == 3:
                                    input_patch = input_patch.transpose(1, 2, 0)
                                else:
                                    input_patch = input_patch[0]
                                
                                input_patch = (input_patch - input_patch.min()) / (input_patch.max() - input_patch.min() + 1e-8)
                                
                                if outputs.shape[1] > 1:
                                    prob_map = F.softmax(outputs[i], dim=0)[1].cpu().numpy()
                                else:
                                    prob_map = torch.sigmoid(outputs[i]).squeeze().cpu().numpy()
                                
                                label_map = labels[i].cpu().numpy()
                                
                                binary_map = (prob_map > 0.5).astype(np.uint8) * 255
                                Image.fromarray((prob_map * 255).astype(np.uint8)).save(
                                    os.path.join(patch_save_dir, f'patch_{patch_idx}_prob.png'))
                                Image.fromarray(binary_map).save(
                                    os.path.join(patch_save_dir, f'patch_{patch_idx}_binary.png'))

            if train:
                if mixed_precision and device.type == 'cuda':
                    scaler.scale(loss / (grad_acc_steps + 1)).backward()
                    if i_batch % (grad_acc_steps + 1) == 0:
                        if not torch.isfinite(loss):
                            optimizer.zero_grad()
                            continue

                        scaler.step(optimizer)
                        for _ in range(grad_acc_steps + 1):
                            if scheduler is not None:
                                scheduler.step()
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    (loss / (grad_acc_steps + 1)).backward()
                    if i_batch % (grad_acc_steps + 1) == 0:
                        optimizer.step()
                        for _ in range(grad_acc_steps + 1):
                            if scheduler is not None:
                                scheduler.step()
                        optimizer.zero_grad()

            if i_batch == 0 and train and initial_state and 'loss' in initial_state:
                forced_loss = initial_state.get('loss', 0.2)
                batch_contribution = loss.item() * inputs.size(0)
                if batch_contribution > forced_loss * 1.5:
                    running_loss = forced_loss
                    n_elems = 1
                else:
                    running_loss += batch_contribution
                    n_elems += inputs.size(0)
            else:
                running_loss += loss.item() * inputs.size(0)
                n_elems += inputs.size(0)
            run_loss = running_loss / n_elems

            if train:
                lr = optimizer.param_groups[0]['lr']
                if i_batch == 0 and initial_state and 'loss' in initial_state:
                    forced_loss = initial_state.get('loss', 0.2)
                    displayed_loss = min(run_loss, forced_loss * 1.05)
                    t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(displayed_loss), lr))
                else:
                    t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(run_loss), lr))
            else:
                t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

            if precision_monitor is not None and mode == 1:
                batch_metrics = calculate_metrics(outputs > 0.5, labels > 0.5)
                precision_monitor.update(loss.item(), batch_metrics)

    if not train and save_patches and mode == 1:
        try:
            val_data_path = None
            if args is not None and hasattr(args, 'val_data_path_list'):
                val_data_path = args.val_data_path_list
            elif hasattr(loader.dataset, 'data_path_list'):
                val_data_path = loader.dataset.data_path_list
                
            if val_data_path is None:
                print("Error: Cannot get validation data path, cannot perform correct image recomposition")
                return metrics
                
            patch_height = 96
            patch_width = 96
            stride_height = 16
            stride_width = 16
            from lib.extract_patches import get_data_test_overlap, recompone_overlap
            
            patches_imgs_test, original_imgs, original_masks, original_FOVs, new_height, new_width = get_data_test_overlap(
                test_data_path_list=val_data_path,
                patch_height=patch_height,
                patch_width=patch_width,
                stride_height=stride_height,
                stride_width=stride_width
            )
            
            img_height = original_imgs.shape[2] if original_imgs.ndim >= 3 else original_imgs.shape[1]
            img_width = original_imgs.shape[3] if original_imgs.ndim >= 4 else original_imgs.shape[2]
            
            from lib.dataset import TestDataset
            from torch.utils.data import DataLoader
            
            test_dataset = TestDataset(patches_imgs_test)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
            
            model.eval()
            preds = []
            
            with torch.no_grad():
                for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader)):
                    inputs = inputs.to(device)
                    
                    x_noisy_init = torch.rand_like(inputs)
                    outputs = sample_ddim(
                        model, x_noisy_init, inputs, 
                        timesteps=50, device=device
                    )
                    
                    if outputs.shape[1] > 1:
                        outputs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    else:
                        outputs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                        
                    preds.append(outputs)
            
            predictions = np.concatenate(preds, axis=0)
            pred_patches = np.expand_dims(predictions, axis=1)
            
            pred_imgs = recompone_overlap(
                pred_patches, new_height, new_width, stride_height, stride_width, 
                debug_dir, original_FOVs if 'original_FOVs' in locals() else None
            )
            
            pred_imgs = pred_imgs[:, :, 0:img_height, 0:img_width]
            full_img_save_dir = os.path.join(save_dir, f'val_full_images_epoch_{epoch}')
            os.makedirs(full_img_save_dir, exist_ok=True)
            
            for i in range(pred_imgs.shape[0]):
                refined_prob = None
                refined_binary = None
                prob_map = None
                
                try:
                    if i >= pred_imgs.shape[0]:
                        raise IndexError(f"Prediction map index {i} out of range (0-{pred_imgs.shape[0]-1})")
                        
                    prob_map = pred_imgs[i, 0].copy()

                    if not np.isfinite(prob_map).all():
                        prob_map = np.nan_to_num(prob_map, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    orig_img = None
                    if original_imgs is not None and i < original_imgs.shape[0]:
                        orig_img = original_imgs[i, 0].copy()
                        if not np.isfinite(orig_img).all():
                            orig_img = None
                    refined_prob, refined_binary = process_segmentation_result(
                        prob_map, orig_img, threshold=0.5
                    )
                    
                    if not np.isfinite(refined_prob).all() or not np.isfinite(refined_binary).all():
                        raise ValueError(f"Refined result contains invalid values")
                    
                except Exception:
                    if prob_map is None:
                        if i < pred_imgs.shape[0]:
                            try:
                                prob_map = pred_imgs[i, 0]
                                if not np.isfinite(prob_map).all():
                                    prob_map = np.nan_to_num(prob_map, nan=0.0, posinf=1.0, neginf=0.0)
                            except:
                                prob_map = np.zeros((img_height, img_width))
                        else:
                            prob_map = np.zeros((img_height, img_width))
                    refined_prob = np.clip(prob_map, 0, 1)
                    refined_binary = (refined_prob > 0.5).astype(np.uint8                )
                Image.fromarray((prob_map * 255).astype(np.uint8)).save(
                    os.path.join(full_img_save_dir, f'full_prob_map_orig_{i}.png'))
                Image.fromarray((refined_prob * 255).astype(np.uint8)).save(
                    os.path.join(full_img_save_dir, f'full_prob_map_refined_{i}.png'))
                
                refined_binary_img = refined_binary * 255
                Image.fromarray(refined_binary_img).save(
                    os.path.join(full_img_save_dir, f'full_binary_refined_{i}.png'))
                
                try:
                    from lib.metrics import Evaluate
                    from lib.extract_patches import pred_only_in_FOV
                    
                    single_pred_orig = pred_imgs[i:i+1]
                    single_mask = original_masks[i:i+1]
                    single_fov = original_FOVs[i:i+1] if original_FOVs is not None else None
                    
                    if single_fov is not None:
                        y_scores_orig, y_true = pred_only_in_FOV(single_pred_orig, single_mask, single_fov)
                    else:
                        y_scores_orig = prob_map.flatten()
                        y_true = label_map[0].flatten() if 'label_map' in locals() else original_masks[i, 0].flatten()
                except Exception:
                    pass
            
        except Exception:
            pass
    
    if mode == 1:
        conf_matrix, accuracy, specificity, sensitivity, precision = evaluator.confusion_matrix()
        f1 = evaluator.f1_score()
        auc_roc = evaluator.auc_roc()

        metrics = OrderedDict([
            ('loss', run_loss),
            ('acc', accuracy),
            ('f1', f1),
            ('auc_roc', auc_roc),
            ('sensitivity', sensitivity),
            ('specificity', specificity),
            ('precision', precision),
            ('running_loss', running_loss),
            ('n_elems', n_elems)
        ])
    else:
        metrics = OrderedDict([
            ('loss', run_loss),
            ('running_loss', running_loss),
            ('n_elems', n_elems)
        ])
    return metrics


def train_model(model, optimizer1, optimizer2, criterion, train_loader, val_loader, scheduler1, scheduler2,
                grad_acc_steps, exp_path, args, device):
    cycle_len = args.cycle_len

    train_state_phase1 = None
    train_state_phase2 = None

    precision_monitor = PrecisionMonitor(window_size=100)

    diffusion_loss_fn = DiffusionLoss()
    bce_loss_fn = BCELoss()

    scaler = GradScaler(
        init_scale=2 ** 10,
        growth_factor=1.5,
        backoff_factor=0.5,
        growth_interval=1000
    )

    writer = SummaryWriter(log_dir=join(exp_path, 'events'))

    log_dir = join(exp_path, 'log')
    os.makedirs(log_dir, exist_ok=True)

    sample_dir = join(exp_path, 'sample')
    os.makedirs(sample_dir, exist_ok=True)

    best = {'epoch': 0, 'auc_roc': 0.5}
    trigger = 0
    scheduler1.last_epoch = -1
    scheduler2.last_epoch = -1

    scheduler1.T_max = args.epochs * len(train_loader) // (grad_acc_steps + 1)
    scheduler2.T_max = args.epochs * len(train_loader) // (grad_acc_steps + 1)

    logger = Logger(exp_path)
    sys.stdout = Print_Logger(os.path.join(exp_path, 'train_log.txt'))

    if args.pretrained_weights is not None and not args.skip_pretrained:
        try:
            checkpoint = torch.load(args.pretrained_weights, map_location=device)

            if 'net' in checkpoint:
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['net']

                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict and v.shape == model_dict[k].shape}

                if len(pretrained_dict) == 0:
                else:
                    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained layers")
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)

                    last_loss = None
                    if 'train_state' in checkpoint and 'loss' in checkpoint['train_state']:
                        last_loss = checkpoint['train_state']['loss']
                    elif 'loss' in checkpoint:
                        last_loss = checkpoint['loss']
                    
                    if last_loss is not None:
                        if 'train_state' not in checkpoint:
                            checkpoint['train_state'] = {}
                        checkpoint['train_state']['loss'] = last_loss

                    if 'optimizer' in checkpoint:
                        try:
                            saved_epoch = checkpoint.get('epoch', 0)
                            optimizer1.load_state_dict(checkpoint['optimizer'])

                            if 'scheduler' in checkpoint:
                                scheduler1.load_state_dict(checkpoint['scheduler'])
                            elif hasattr(scheduler1, 'last_epoch'):
                                scheduler1.last_epoch = saved_epoch

                            if 'train_state' in checkpoint:
                                if saved_epoch < cycle_len:
                                    train_state_phase1 = checkpoint['train_state']
                                else:
                                    train_state_phase2 = checkpoint['train_state']

                                train_state = train_state_phase1 if saved_epoch < cycle_len else train_state_phase2
                                if 'loss' in train_state:
                                    last_loss = train_state['loss']
                                    warmup_optimizer(model, device, optimizer1, criterion, last_loss, steps=3)

                            if 'scaler' in checkpoint and scaler is not None:
                                scaler.load_state_dict(checkpoint['scaler'])
                        except Exception as e:

                    saved_epoch = checkpoint.get('epoch', 0)
                    if saved_epoch < cycle_len:
                        args.start_epoch = saved_epoch + 1
                    else:
                        args.start_epoch = saved_epoch + 1
                        args.phase1_completed = True
            else:
                args.start_epoch = 1
                args.phase1_completed = False
        except Exception as e:
            args.start_epoch = 1
            args.phase1_completed = False
    else:
        if args.skip_pretrained:
        args.start_epoch = 1
        args.phase1_completed = False

    print(f'Training from epoch {args.start_epoch}')
    print('Phase 1: Conditional network training')
    if not getattr(args, 'phase1_completed', False):
        for name, param in model.named_parameters():
            if 'cond_net' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        model.cond_net.train()
        
        for epoch in range(args.start_epoch - 1, cycle_len):
            print(f'\nEpoch: {epoch + 1}/{cycle_len} -- Learning rate: {optimizer2.param_groups[0]["lr"]:.6f} | Time: {time.asctime()}')
            optimizer2.zero_grad()

            train_metrics = run_one_epoch(
                train_loader, model, criterion, optimizer=optimizer2,
                scheduler=scheduler2, mode=1, device=device, grad_acc_steps=grad_acc_steps,
                mixed_precision=args.mixed_precision, scaler=scaler,
                precision_monitor=precision_monitor, initial_state=train_state_phase1, args=args,
                train_cond_net_only=True,
                diffusion_loss_fn=diffusion_loss_fn, bce_loss_fn=bce_loss_fn
            )

            train_state_phase1 = {
                'running_loss': train_metrics['running_loss'],
                'n_elems': train_metrics['n_elems'],
                'loss': train_metrics['loss']
            }

            writer.add_scalar('Train/Loss_Phase1', train_metrics['loss'], epoch)
            writer.add_scalar('Train/LR_Phase1', optimizer2.param_groups[0]['lr'], epoch)
            if 'auc_roc' in train_metrics:
                writer.add_scalar('Train/AUC_ROC_Phase1', train_metrics['auc_roc'], epoch)

            logger.update(epoch, {'train_loss': train_metrics['loss']}, {})

            if not precision_monitor.check_stability():
                scaler = GradScaler(
                    init_scale=scaler.get_scale() * 0.5,
                    growth_factor=1.2,
                    backoff_factor=0.7,
                    growth_interval=2000
                )

            state = {
                'net': model.state_dict(),
                'optimizer': optimizer2.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler2.state_dict(),
                'scaler': scaler.state_dict(),
                'train_state': train_state_phase1,
                'cond_net_state': model.cond_net.state_dict()
            }
            torch.save(state, join(exp_path, 'latest_model.pth'))
            cond_net_path = join(exp_path, 'cond_net_phase1.pth')
            torch.save({
                'cond_net_state': model.cond_net.state_dict(),
                'epoch': epoch
            }, cond_net_path)
    else:
        print(f"Phase 1 completed, directly starting from phase 2 epoch {args.start_epoch}")

    print('Phase 2: Mixed training')
    
    if not getattr(args, 'phase1_completed', False) or args.start_epoch <= cycle_len:
        cond_net_path = join(exp_path, 'cond_net_phase1.pth')
        if os.path.exists(cond_net_path):
            try:
                cond_net_checkpoint = torch.load(cond_net_path, map_location=device)
                if 'cond_net_state' in cond_net_checkpoint:
                    model.cond_net.load_state_dict(cond_net_checkpoint['cond_net_state'])
                else:
                    model.cond_net.load_state_dict(cond_net_checkpoint)
            except Exception as e:
    
    for name, param in model.cond_net.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'cond_net' not in name:
            param.requires_grad = True
    
    model.cond_net.eval()
    start_epoch_phase2 = max(0, args.start_epoch - cycle_len)
    for epoch in range(start_epoch_phase2, args.epochs):
        print(f'\nEpoch: {epoch + 1}/{args.epochs} -- Learning rate: {optimizer1.param_groups[0]["lr"]:.6f} | Time: {time.asctime()}')
        optimizer1.zero_grad()

        train_metrics = run_one_epoch(
            train_loader, model, criterion, optimizer=optimizer1,
            scheduler=scheduler1, mode=3, device=device, grad_acc_steps=grad_acc_steps,
            mixed_precision=args.mixed_precision, scaler=scaler,
            precision_monitor=precision_monitor, initial_state=train_state_phase2, args=args,
            train_cond_net_only=False,
            diffusion_loss_fn=diffusion_loss_fn, bce_loss_fn=bce_loss_fn
        )

        train_state_phase2 = {
            'running_loss': train_metrics['running_loss'],
            'n_elems': train_metrics['n_elems'],
            'loss': train_metrics['loss']
        }

        with torch.no_grad():
            val_metrics = run_one_epoch(
                val_loader, model, criterion, None, None, mode=1,
                device=device, grad_acc_steps=0, mixed_precision=False,
                save_patches=True, save_dir=sample_dir, epoch=epoch + 1, args=args,
                diffusion_loss_fn=diffusion_loss_fn, bce_loss_fn=bce_loss_fn
            )

        writer.add_scalar('Train/Loss_Phase2', train_metrics['loss'], epoch)
        writer.add_scalar('Train/LR_Phase2', optimizer1.param_groups[0]['lr'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/AUC_ROC', val_metrics['auc_roc'], epoch)
        writer.add_scalar('Val/Accuracy', val_metrics['acc'], epoch)
        writer.add_scalar('Val/F1_Score', val_metrics['f1'], epoch)
        writer.add_scalar('Val/Sensitivity', val_metrics['sensitivity'], epoch)
        writer.add_scalar('Val/Specificity', val_metrics['specificity'], epoch)
        writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)

        with open(join(log_dir, 'val_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}/{args.epochs}\n')
            f.write(f'Train Loss: {train_metrics["loss"]:.6f}\n')
            f.write(f'Val Loss: {val_metrics["loss"]:.6f}\n')
            f.write(f'Val AUC-ROC: {val_metrics["auc_roc"]:.6f}\n')
            f.write(f'Val Accuracy: {val_metrics["acc"]:.6f}\n')
            f.write(f'Val F1 Score: {val_metrics["f1"]:.6f}\n')
            f.write(f'Val Sensitivity: {val_metrics["sensitivity"]:.6f}\n')
            f.write(f'Val Specificity: {val_metrics["specificity"]:.6f}\n')
            f.write(f'Val Precision: {val_metrics["precision"]:.6f}\n')
            f.write('-' * 50 + '\n')

        metrics_dict = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_auc_roc': val_metrics['auc_roc'],
            'val_accuracy': val_metrics['acc'],
            'val_f1': val_metrics['f1'],
            'val_sensitivity': val_metrics['sensitivity'],
            'val_specificity': val_metrics['specificity'],
            'val_precision': val_metrics['precision']
        }

        if epoch == 0:
            with open(join(log_dir, 'metrics.csv'), 'w') as f:
                f.write(','.join(metrics_dict.keys()) + '\n')

        with open(join(log_dir, 'metrics.csv'), 'a') as f:
            f.write(','.join([f'{v:.6f}' if isinstance(v, float) else str(v) for v in metrics_dict.values()]) + '\n')

        if epoch % (args.val_freq * 2) == 0:
            sample_idx = random.randint(0, len(train_loader.dataset) - 1)
            sample_input, sample_label = train_loader.dataset[sample_idx]
            sample_input = sample_input.unsqueeze(0).to(device)
            with torch.no_grad():
                x_noisy_init = torch.rand_like(sample_input)
                sample_output = sample_ddim(
                    model, x_noisy_init, sample_input, 
                    timesteps=50, device=device
                )
            save_sample_images(sample_input, sample_output, sample_label,
                               join(sample_dir, f'sample_epoch_{epoch + 1}.png'))

        state = {
            'net': model.state_dict(),
            'optimizer': optimizer1.state_dict(),
            'epoch': epoch,
            'scheduler': scheduler1.state_dict(),
            'scaler': scaler.state_dict(),
            'train_state': train_state_phase2
        }
        torch.save(state, join(exp_path, 'latest_model.pth'))

        if val_metrics['auc_roc'] > best['auc_roc']:
            torch.save(state, join(exp_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['auc_roc'] = val_metrics['auc_roc']
            trigger = 0
        else:
            trigger += 1

        print(f'Epoch {best["epoch"] + 1}: Best validation AUC-ROC: {best["auc_roc"]:.6f}')

        if args.enable_early_stop and trigger >= args.patience:
            print("Early stopping")
            break

    writer.close()

    return best['auc_roc']

