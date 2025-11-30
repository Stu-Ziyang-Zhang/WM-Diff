import argparse
import os
import yaml
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--outf', default='./xiao_rong', help='Output directory for trained models')
    parser.add_argument('--save', default='test', help='Experiment name')
    parser.add_argument('--snapshot_path', default='./checkpoints', help='Checkpoint save path')
    parser.add_argument('--save_path', default='test', help='Save path name, default is date time')

    parser.add_argument('--train_data_path', default='./data/train', help='Training data folder path')
    parser.add_argument('--train_data_path_list', default='./prepare_dataset/data_path_list/CHASEDB1/train.txt', help='Training data path list')
    parser.add_argument('--test_data_path_list', default='./prepare_dataset/data_path_list/CHASEDB1/test.txt', help='Test data path list')

    parser.add_argument('--train_patch_height', default=64, type=int, help='Training patch height')
    parser.add_argument('--train_patch_width', default=64, type=int, help='Training patch width')
    parser.add_argument('--N_patches', default=150000, type=int, help='Number of training image patches')
    parser.add_argument('--inside_FOV', default='center', help='Select [not,center,all]')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='Validation set ratio in training set')
    parser.add_argument('--sample_visualization', default=True, type=bool, help='Training sample visualization')
    parser.add_argument('--im_size', default='512', help='Image size, e.g., 512 or 512,384')
    parser.add_argument('--test_patch_height', default=96, type=int, help='Test patch height')
    parser.add_argument('--test_patch_width', default=96, type=int, help='Test patch width')
    parser.add_argument('--stride_height', default=16, type=int, help='Sliding window height stride')
    parser.add_argument('--stride_width', default=16, type=int, help='Sliding window width stride')
    
    parser.add_argument('--model_name', default='diffusion_cond_unet', help='Model architecture name')
    parser.add_argument('--in_channels', default=1, type=int, help='Model input channels')
    parser.add_argument('--classes', default=2, type=int, help='Classification model output channels')
    parser.add_argument('--out_channels', default=1, type=int, help='Segmentation model output channels')
    parser.add_argument('--dim', default=32, type=int, help='Diffusion UNet initial channels')
    parser.add_argument('--dim_mults', default=[1, 2, 4, 8], type=list, help='Diffusion UNet channel multipliers')
    parser.add_argument('--base_channels', default=32, type=int, help='Conditional network base channels')
    parser.add_argument('--growth_rate', default=32, type=int, help='Dense block growth rate')

    parser.add_argument('--epochs', default=100, type=int, help='Total training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--patience', default=15, type=int, help='Early stopping patience')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='Initial learning rate')
    parser.add_argument('--min_lr', default=0.00005, type=float, help='Minimum learning rate')
    parser.add_argument('--max_lr', default=0.0005, type=float, help='Maximum learning rate')
    parser.add_argument('--val_on_test', default=False, type=bool, help='Validate on test set')
    parser.add_argument('--val_freq', default=1, type=int, help='Validation frequency: validate every N epochs')
    parser.add_argument('--enable_early_stop', default=True, type=bool, help='Enable early stopping')
    parser.add_argument('--seed', default=2021, type=int, help='Random seed')
    parser.add_argument('--cycle_len', default=5, type=int, help='Cosine annealing cycle length')
    parser.add_argument('--gradient_accumulation_steps', default=0, type=int, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', default=False, type=bool, help='Use mixed precision training')
    parser.add_argument('--do_not_save', default=False, type=bool, help='Do not save anything')

    parser.add_argument('--start_epoch', default=1, type=int, help='Start epoch')
    parser.add_argument('--pretrained_weights', default=None, help='Pretrained weights path')
    parser.add_argument('--skip_pretrained', default=False, type=bool, help='Skip loading pretrained weights')

    parser.add_argument('--cuda', default=True, type=bool, help='Use GPU')
    parser.add_argument('--device', default='cuda:0', help='Training device, e.g., cuda:0 or cpu')
    parser.add_argument('--num_workers', default=8, type=int, help='Data loader worker threads')
                       
    parser.add_argument('--tensorboard', default=True, type=bool, help='Use TensorBoard to log training')
    
    parser.add_argument('--fold', default=10, type=int, help='Current training fold (for cross-validation)')

    parser.add_argument('--test_only', action='store_true', help='Run test only, no training')
    parser.add_argument('--test_save_dir', default='diffusion_test_results', help='Test results save directory')

    parser.add_argument('--inference_mode', action='store_true', help='Enable inference mode, test with pretrained model')
    parser.add_argument('--model_path', type=str, help='Model path for inference')
    parser.add_argument('--output', default='output', type=str, help='Inference output directory')

    args = parser.parse_args()
    return args

def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist!")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(file_content)
        elif config_path.endswith('.json'):
                config = json.loads(file_content)
            else:
                try:
                config = yaml.safe_load(file_content)
            except:
                try:
                    config = json.loads(file_content)
                except:
                    raise ValueError(f"Cannot parse config file: {config_path}, supported formats: YAML, JSON")
    
    if not config:
        raise ValueError(f"Config file is empty or format error: {config_path}")
    
    class ConfigNamespace:
        pass
    
    args = ConfigNamespace()
    
    for key, value in config.items():
        setattr(args, key, value)
    
    param_mapping = {
        'epochs': ['epoch_num', 'N_epochs'],
        'learning_rate': ['lr', 'max_lr'],
        'patience': ['early-stop'],
        'base_channels': ['base_ch'],
        'gradient_accumulation_steps': ['grad_acc_steps'],
        'pretrained_weights': ['pre_trained']
    }
    
    for new_param, old_params in param_mapping.items():
        for old_param in old_params:
            if hasattr(args, old_param) and not hasattr(args, new_param):
                setattr(args, new_param, getattr(args, old_param))
    
    return args

def create_default_config(save_path='config.yaml'):
    args = parse_args()
    config_dict = vars(args)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

def calculate_metrics(pred, target):
    if pred.dim() == 4:
        if pred.shape[1] > 1:
            if pred.shape[1] == 2:
                pred = pred[:, 1]
            else:
                pred = torch.argmax(pred, dim=1)
        else:
            pred = pred.squeeze(1)
    
    if target.dim() == 4:
        if target.shape[1] == 1:
            target = target.squeeze(1)
        else:
            target = target[:, 1] if target.shape[1] > 1 else target[:, 0]
    
    if pred.shape != target.shape:
        if pred.dim() == 3 and target.dim() == 3:
            if pred.shape[1:] != target.shape[1:]:
                target = torch.nn.functional.interpolate(
                    target.unsqueeze(1),
                    size=pred.shape[1:],
                    mode='nearest'
                ).squeeze(1)
        min_batch = min(pred.shape[0], target.shape[0])
        pred = pred[:min_batch]
        target = target[:min_batch]
    
    pred = pred.bool()
    target = target.bool()
    
    pred_flat = pred.cpu().view(-1)
    target_flat = target.cpu().view(-1)
    
    if pred_flat.shape[0] != target_flat.shape[0]:
        min_size = min(pred_flat.shape[0], target_flat.shape[0])
        pred_flat = pred_flat[:min_size]
        target_flat = target_flat[:min_size]
    
    tp = torch.logical_and(pred_flat, target_flat).sum().float().item()
    fp = torch.logical_and(pred_flat, torch.logical_not(target_flat)).sum().float().item()
    fn = torch.logical_and(torch.logical_not(pred_flat), target_flat).sum().float().item()
    tn = torch.logical_and(torch.logical_not(pred_flat), torch.logical_not(target_flat)).sum().float().item()
    
    smooth = 1e-5
    
    dice = (2 * tp) / (2 * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = sensitivity = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    create_default_config()
