import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys
import os
import tempfile
from os.path import join
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from PIL import Image

from config import parse_args, load_config
from lib.common import setup_seed, count_parameters
from function import get_dataloaderV2
from diffusion_framework import DiffusionCondUNet
from training import train_model, run_one_epoch
from lib.sampling import sample_ddim
from lib.losses.loss import BCEWithLogitsLoss
from lib.metrics import Evaluate
from lib.visualize import save_img

DEFAULT_TRAIN_PATHS = [
    './prepare_dataset/data_path_list/DRIVE/train.txt',
    './prepare_dataset/data_path_list/STARE_aug/train.txt',
    './prepare_dataset/data_path_list/CHASEDB1/train.txt'
]
DEFAULT_TEST_PATHS = [
    './prepare_dataset/data_path_list/DRIVE/test.txt',
    './prepare_dataset/data_path_list/STARE_aug/test.txt',
    './prepare_dataset/data_path_list/CHASEDB1/test.txt'
]

def find_default_path(path_list, default_paths):
    """Find existing default path"""
    for path in default_paths:
        if os.path.exists(path):
            return path
    return default_paths[0] if default_paths else None

def normalize_path_list(path_list, default_paths):
    """Normalize path list: create temp file if list, check existence if str"""
    if isinstance(path_list, str):
        if path_list.endswith('.txt') and os.path.exists(path_list):
            return path_list, None
        return find_default_path(path_list, default_paths), None
    
    elif isinstance(path_list, list):
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        if len(path_list) >= 3 and all(' ' in path for path in path_list[:3]):
            for path in path_list:
                temp_file.write(f"{path}\n")
        elif len(path_list) >= 3:
            temp_file.write(f"{path_list[0]} {path_list[1]} {path_list[2]}\n")
        else:
            for path in path_list:
                temp_file.write(f"{path}\n")
        temp_file.close()
        return temp_file.name, temp_file.name
    
    return path_list, None

def main():
    setup_seed(2023)
    try:
        if os.path.exists('config.yaml'):
            args = load_config('config.yaml')
        else:
            args = parse_args()

        args.skip_pretrained = False
        if len(sys.argv) > 1:
            if '--skip-pretrained' in sys.argv:
                args.skip_pretrained = True
            elif '--pretrained-weights' in sys.argv:
                idx = sys.argv.index('--pretrained-weights')
                if idx + 1 < len(sys.argv):
                    args.pretrained_weights = sys.argv[idx + 1]
            elif '--test-only' in sys.argv:
                args.test_only = True
        
        if not hasattr(args, 'pretrained_weights') or args.pretrained_weights is None:
            pretrained_dir = 'experiments/pretrained'
            if os.path.exists(pretrained_dir):
                pretrained_files = [f for f in os.listdir(pretrained_dir) if f.endswith('.pth')]
                if pretrained_files:
                    pretrained_files.sort(key=lambda x: os.path.getmtime(os.path.join(pretrained_dir, x)))
                    args.pretrained_weights = os.path.join(pretrained_dir, pretrained_files[-1])
    except Exception:
        args = parse_args()
    
    if not hasattr(args, 'train_data_path_list') or args.train_data_path_list is None:
        args.train_data_path_list = DEFAULT_TRAIN_PATHS[0]
    args.train_data_path_list, temp_train_path = normalize_path_list(args.train_data_path_list, DEFAULT_TRAIN_PATHS)
    
    if not hasattr(args, 'val_data_path_list') or args.val_data_path_list is None:
        if hasattr(args, 'val_on_test') and args.val_on_test and hasattr(args, 'test_data_path_list'):
                                args.val_data_path_list = args.test_data_path_list
                    else:
            args.val_data_path_list = None

    if args.val_data_path_list:
        args.val_data_path_list, _ = normalize_path_list(args.val_data_path_list, DEFAULT_TEST_PATHS)

    if not hasattr(args, 'test_data_path_list') and args.val_data_path_list:
        args.test_data_path_list = args.val_data_path_list

    if args.device.startswith("cuda"):
        if ":" in args.device:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":", 1)[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            cudnn.benchmark = True
            cudnn.deterministic = True
    else:
        device = torch.device(args.device)
    if not args.do_not_save:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_path = join(args.outf, save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(join(experiment_path, 'events'), exist_ok=True)
        os.makedirs(join(experiment_path, 'log'), exist_ok=True)
        os.makedirs(join(experiment_path, 'sample'), exist_ok=True)
        config_file_path = join(experiment_path, 'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else:
        experiment_path = None

    has_separate_val = hasattr(args, 'val_data_path_list') and args.val_data_path_list is not None
    if not has_separate_val:
        if not hasattr(args, 'val_ratio') or args.val_ratio <= 0 or args.val_ratio >= 1:
            args.val_ratio = 0.1
        train_loader, val_loader = get_dataloaderV2(args)

    if temp_train_path:
        try:
            os.unlink(temp_train_path)
        except Exception:
            pass

    use_checkpoint = getattr(args, 'use_checkpoint', False)
    
    model = DiffusionCondUNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        dim=args.dim,
        dim_mults=args.dim_mults,
        base_ch=args.base_channels,
        growth_rate=args.growth_rate,
        use_checkpoint=use_checkpoint
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")
    criterion = BCEWithLogitsLoss()

    if hasattr(args, 'test_only') and args.test_only:
        if not hasattr(args, 'pretrained_weights') or args.pretrained_weights is None:
            print("Error: Pretrained weights required in test-only mode")
            return

        if not os.path.exists(args.pretrained_weights) or os.path.getsize(args.pretrained_weights) == 0:
            alt_weights = "diffusion/DRIVE_2/best_model.pth"
            if os.path.exists(alt_weights) and os.path.getsize(alt_weights) > 0:
                args.pretrained_weights = alt_weights
            else:
                print("Error: Pretrained weights file not found")
                return

        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        if 'net' in checkpoint:
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['net'].items()
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained layers")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            print("Error: Invalid checkpoint format")
            return

        test_output_dir = os.path.join(args.outf, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(test_output_dir, exist_ok=True)
        test_data_path = args.test_data_path_list if hasattr(args, 'test_data_path_list') else args.val_data_path_list
        temp_test_path, _ = normalize_path_list(test_data_path, DEFAULT_TEST_PATHS)
        
        original_train_path = args.train_data_path_list
        args.train_data_path_list = temp_test_path
        _, test_loader = get_dataloaderV2(args)
        args.train_data_path_list = original_train_path

        if temp_test_path != test_data_path:
            try:
                os.unlink(temp_test_path)
            except Exception:
                pass

        from lib.losses.loss import DiffusionLoss, BCELoss
        model.eval()
        with torch.no_grad():
            loss_value = None
            if 'train_state' in checkpoint and 'loss' in checkpoint['train_state']:
                loss_value = checkpoint['train_state']['loss']
            elif 'loss' in checkpoint:
                loss_value = checkpoint['loss']
            
            test_initial_state = None
            if loss_value is not None:
                test_initial_state = {'loss': loss_value, 'running_loss': loss_value, 'n_elems': 1}

            test_metrics = run_one_epoch(
                test_loader, model, criterion, None, None, mode=1,
                device=device, grad_acc_steps=0, mixed_precision=False,
                initial_state=test_initial_state, args=args,
                diffusion_loss_fn=DiffusionLoss(), bce_loss_fn=BCELoss()
            )

        print(f"Test Results - Loss: {test_metrics['loss']:.6f}, AUC-ROC: {test_metrics['auc_roc']:.6f}, "
              f"Accuracy: {test_metrics['acc']:.6f}, F1: {test_metrics['f1']:.6f}, "
              f"Sensitivity: {test_metrics['sensitivity']:.6f}, Specificity: {test_metrics['specificity']:.6f}, "
              f"Precision: {test_metrics['precision']:.6f}")

        metrics_file = os.path.join(test_output_dir, 'test_metrics.csv')
        with open(metrics_file, 'w') as f:
            f.write('metric,value\n')
            for k, v in test_metrics.items():
                f.write(f'{k},{v:.6f}\n')


        result_dir = os.path.join(test_output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        debug_dir = os.path.join(test_output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        from lib.extract_patches import recompone_overlap

        batch_size = 8
        test_data_path = args.test_data_path_list if hasattr(args, 'test_data_path_list') else args.val_data_path_list
        temp_test_path, _ = normalize_path_list(test_data_path, DEFAULT_TEST_PATHS)
        
        try:
            from lib.extract_patches import get_data_test_overlap
            patches_imgs_test, test_imgs_original, test_masks, test_FOVs, new_height, new_width = get_data_test_overlap(
                test_data_path_list=temp_test_path,
                patch_height=96,
                patch_width=96,
                stride_height=16,
                stride_width=16
            )
        finally:
            if temp_test_path != test_data_path:
                try:
                    os.unlink(temp_test_path)
                except Exception:
                    pass

        test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(patches_imgs_test).float(),
            torch.zeros(patches_imgs_test.shape[0])
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        )

        pred_patches = []
        model.eval()

        patches_dir = os.path.join(test_output_dir, 'patches')
        os.makedirs(patches_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(tqdm(test_loader)):
                inputs = inputs.to(device)
                x_noisy_init = torch.rand_like(inputs)
                outputs = sample_ddim(
                    model, x_noisy_init, inputs,
                    timesteps=50, device=device
                )

                if outputs.shape[1] > 1:
                    probs = F.softmax(outputs, dim=1)
                    vessel_probs = probs[:, 1].cpu().numpy()
                else:
                    vessel_probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

                for i in range(vessel_probs.shape[0]):
                    patch_idx = batch_idx * batch_size + i
                    if patch_idx < len(patches_imgs_test):
                        prob_map = vessel_probs[i]
                        binary_map = (prob_map > 0.5).astype(np.uint8) * 255
                        Image.fromarray((prob_map * 255).astype(np.uint8)).save(
                            os.path.join(patches_dir, f'patch_{patch_idx}_prob.png'))
                        Image.fromarray(binary_map).save(
                            os.path.join(patches_dir, f'patch_{patch_idx}_binary.png'))

                pred_patches.append(vessel_probs)

        pred_patches = np.concatenate(pred_patches, axis=0)

        if len(pred_patches.shape) == 3:
            pred_patches = np.expand_dims(pred_patches, axis=1)

        pred_imgs = recompone_overlap(
            pred_patches, new_height, new_width, 16, 16, debug_dir, test_FOVs
        )

        orig_height = test_imgs_original.shape[2]
        orig_width = test_imgs_original.shape[3]
        pred_imgs = pred_imgs[:, :, 0:orig_height, 0:orig_width]

        if test_FOVs is not None:
            from lib.extract_patches import kill_border
            kill_border(pred_imgs, test_FOVs)

        eval_dir = os.path.join(test_output_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

        all_metrics = {
            'auc_roc': [],
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1_score': [],
            'best_threshold': []
        }

        for i in range(pred_imgs.shape[0]):
            prob_map = pred_imgs[i, 0]
            test_mask = test_masks[i, 0] if test_masks.ndim == 4 else test_masks[i]

            Image.fromarray((prob_map * 255).astype(np.uint8)).save(
                os.path.join(result_dir, f'full_prob_map_{i}.png'))

            single_eval_dir = os.path.join(eval_dir, f'image_{i}')
            os.makedirs(single_eval_dir, exist_ok=True)

            if test_FOVs is not None and test_FOVs.shape[0] > i:
                fov_mask = test_FOVs[i, 0] if test_FOVs.ndim == 4 else test_FOVs[i]
                valid_pixels = fov_mask > 0
                flat_preds = prob_map.flatten()[valid_pixels.flatten()]
                flat_labels = test_mask.flatten()[valid_pixels.flatten()]
                fov_prob_map = prob_map[valid_pixels]
                fov_labels = test_mask[valid_pixels]
            else:
                flat_preds = prob_map.flatten()
                flat_labels = test_mask.flatten()
                fov_prob_map = prob_map.flatten()
                fov_labels = test_mask.flatten()

            from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
            fpr, tpr, _ = roc_curve(flat_labels, flat_preds)
            auroc = auc(fpr, tpr)
            all_metrics['auc_roc'].append(auroc)
            best_f1 = 0
            best_threshold = 0.5
            thresholds_to_try = np.linspace(0.1, 0.9, 17)
            all_threshold_metrics = {}

            for thresh in thresholds_to_try:
                binary_preds = (fov_prob_map >= thresh).astype(np.uint8)
                f1 = f1_score(fov_labels, binary_preds)

                accuracy = accuracy_score(fov_labels, binary_preds)
                precision_val = precision_score(fov_labels, binary_preds, zero_division=0)
                recall_val = recall_score(fov_labels, binary_preds, zero_division=0)
                specificity_val = np.sum((binary_preds==0) & (fov_labels==0)) / max(np.sum(fov_labels==0), 1)
                all_threshold_metrics[thresh] = {
                    'f1': f1,
                    'accuracy': accuracy,
                    'sensitivity': recall_val,
                    'specificity': specificity_val,
                    'precision': precision_val,
                }

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh

            all_metrics['best_threshold'].append(best_threshold)
            evaluator = Evaluate()
            evaluator.add_batch(flat_labels, flat_preds)
            _, accuracy, specificity, sensitivity, precision_val = evaluator.confusion_matrix()
            f1 = evaluator.f1_score()

            all_metrics['accuracy'].append(accuracy)
            all_metrics['sensitivity'].append(sensitivity)
            all_metrics['specificity'].append(specificity)
            all_metrics['precision'].append(precision_val)
            all_metrics['f1_score'].append(f1)

            threshold_csv_path = os.path.join(single_eval_dir, 'threshold_metrics.csv')
            with open(threshold_csv_path, 'w') as f:
                f.write('threshold,f1,accuracy,sensitivity,specificity,precision\n')
                for thresh in thresholds_to_try:
                    metrics = all_threshold_metrics[thresh]
                    f.write(f"{thresh:.3f},{metrics['f1']:.6f},{metrics['accuracy']:.6f},{metrics['sensitivity']:.6f},{metrics['specificity']:.6f},{metrics['precision']:.6f}\n")


            with open(os.path.join(single_eval_dir, 'metrics.csv'), 'w') as f:
                f.write('metric,value\n')
                f.write(f'auc_roc,{auroc:.6f}\n')
                f.write(f'accuracy,{accuracy:.6f}\n')
                f.write(f'sensitivity,{sensitivity:.6f}\n')
                f.write(f'specificity,{specificity:.6f}\n')
                f.write(f'precision,{precision_val:.6f}\n')
                f.write(f'f1_score,{f1:.6f}\n')
                f.write(f'best_threshold,{best_threshold:.6f}\n')

                best_thresh_metrics = all_threshold_metrics[best_threshold]
                f.write(f'best_thresh_f1,{best_thresh_metrics["f1"]:.6f}\n')
                f.write(f'best_thresh_accuracy,{best_thresh_metrics["accuracy"]:.6f}\n')
                f.write(f'best_thresh_sensitivity,{best_thresh_metrics["sensitivity"]:.6f}\n')
                f.write(f'best_thresh_specificity,{best_thresh_metrics["specificity"]:.6f}\n')
                f.write(f'best_thresh_precision,{best_thresh_metrics["precision"]:.6f}\n')


            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            for thresh in thresholds:
                binary_pred = (prob_map > thresh).astype(np.uint8) * 255
                Image.fromarray(binary_pred).save(
                    os.path.join(result_dir, f'full_binary_pred_{i}_threshold_{thresh:.2f}.png'))

            binary_pred_best = (prob_map > best_threshold).astype(np.uint8) * 255
            Image.fromarray(binary_pred_best).save(
                os.path.join(result_dir, f'full_binary_pred_{i}_best_threshold_{best_threshold:.2f}.png'))

            from lib.visualize import concat_result
            test_img = test_imgs_original[i, 0] if test_imgs_original.ndim == 4 else test_imgs_original[i]
            binary_pred_viz = (prob_map > best_threshold).astype(np.float32)
            total_img = concat_result(test_img, binary_pred_viz, test_mask)
            save_img(total_img, os.path.join(result_dir, f'comparison_{i}.png'))

        mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
        best_thresh_metrics_all = {'best_f1': [], 'best_accuracy': [], 'best_sensitivity': [], 
                                   'best_specificity': [], 'best_precision': []}
        for i in range(pred_imgs.shape[0]):
            metrics_file = os.path.join(eval_dir, f'image_{i}', 'metrics.csv')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        for line in f:
                            if line.startswith('best_thresh_'):
                                key = line.split(',')[0].replace('best_thresh_', 'best_')
                                if key in best_thresh_metrics_all:
                                    best_thresh_metrics_all[key].append(float(line.split(',')[1]))
                except Exception:
                    pass

        best_thresh_mean = {k: np.mean(v) if v else 0.0 for k, v in best_thresh_metrics_all.items()}
        best_thresh_std = {k: np.std(v) if v else 0.0 for k, v in best_thresh_metrics_all.items()}

        with open(os.path.join(eval_dir, 'average_metrics.csv'), 'w') as f:
            f.write('metric,mean,std\n')
            for k in mean_metrics.keys():
                f.write(f'{k},{mean_metrics[k]:.6f},{std_metrics[k]:.6f}\n')
            for k in best_thresh_mean.keys():
                f.write(f'{k},{best_thresh_mean[k]:.6f},{best_thresh_std[k]:.6f}\n')

        print(f"Evaluation Results - AUROC: {mean_metrics['auc_roc']:.4f}±{std_metrics['auc_roc']:.4f}, "
              f"F1: {mean_metrics['f1_score']:.4f}±{std_metrics['f1_score']:.4f}, "
              f"Accuracy: {mean_metrics['accuracy']:.4f}±{std_metrics['accuracy']:.4f}, "
              f"Sensitivity: {mean_metrics['sensitivity']:.4f}±{std_metrics['sensitivity']:.4f}, "
              f"Specificity: {mean_metrics['specificity']:.4f}±{std_metrics['specificity']:.4f}, "
              f"Precision: {mean_metrics['precision']:.4f}±{std_metrics['precision']:.4f}")
        print(f"Best Threshold Results - F1: {best_thresh_mean['best_f1']:.4f}±{best_thresh_std['best_f1']:.4f}, "
              f"Accuracy: {best_thresh_mean['best_accuracy']:.4f}±{best_thresh_std['best_accuracy']:.4f}, "
              f"Threshold: {mean_metrics['best_threshold']:.4f}±{std_metrics['best_threshold']:.4f}")
        return

    optimizer1 = optim.Adam([
        {'params': model.init_conv.parameters()},
        {'params': model.down_blocks.parameters()},
        {'params': model.mid_block1.parameters()},
        {'params': model.mid_attn.parameters()},
        {'params': model.mid_block2.parameters()},
        {'params': model.up_blocks.parameters()},
        {'params': model.final_conv.parameters()}
    ], lr=args.max_lr)

    optimizer2 = optim.Adam([
        {'params': model.cond_net.parameters()}
    ], lr=args.max_lr)

    if not hasattr(args, 'min_lr'):
        args.min_lr = 0.0002
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=args.epochs * len(train_loader) // (args.gradient_accumulation_steps + 1),
        eta_min=args.min_lr
    )

    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2,
        T_max=args.epochs * len(train_loader) // (args.gradient_accumulation_steps + 1),
        eta_min=args.min_lr
    )


    best_auc_roc = train_model(
        model, optimizer1, optimizer2, criterion, train_loader, val_loader,
        scheduler1, scheduler2, args.gradient_accumulation_steps,
        experiment_path, args, device
    )

    print(f"Training completed. Best validation AUC-ROC: {best_auc_roc:.6f}")


if __name__ == '__main__':
    main()
