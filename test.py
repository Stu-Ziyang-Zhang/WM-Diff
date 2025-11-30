import os
import sys
import torch
import numpy as np
import json
from os.path import join
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score
import cv2
import torch.nn.functional as F

from diffusion_framework import DiffusionCondUNet
from lib.extract_patches import recompone_overlap, get_data_test_overlap
from lib.common import setup_seed
from lib.metrics import Evaluate
from config import parse_args, load_config



class DiffusionTester:
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        self.config_path = join(args.output_dir, "config.cfg") if args.config_file is None else args.config_file
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
        self.model = None
        self.config = None
        
        self.load_config()
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'in_channels': 1,
                'out_channels': 1,
                'classes': 2,
                'dim': 32,
                'dim_mults': [1, 2, 4, 8],
                'base_channels': 32,
                'growth_rate': 32,
                'test_patch_height': 96,
                'test_patch_width': 96,
                'stride_height': 16,
                'stride_width': 16
            }
    
    def load_model(self, model_path=None):
        """Load pretrained model"""
        if model_path is None:
            model_path = join(self.output_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            return False
        
        self.model = DiffusionCondUNet(
            in_channels=self.config.get('in_channels', 1),
            out_channels=self.config.get('out_channels', 1),
            dim=self.config.get('dim', 32),
            dim_mults=self.config.get('dim_mults', [1, 2, 4, 8]),
            base_ch=self.config.get('base_channels', 32),
            growth_rate=self.config.get('growth_rate', 32)
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'net' in checkpoint:
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['net'].items()
                                 if k in model_dict and v.shape == model_dict[k].shape}
                
                if len(pretrained_dict) == 0:
                    return False
                
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                return True
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                return True
            return False
        except Exception:
            return False
    
    def generate_predictions(self, test_loader):
        """Generate predictions"""
        from lib.sampling import sample_ddim
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(test_loader, total=len(test_loader)):
                inputs = inputs.to(self.device)
                x_noisy_init = torch.rand_like(inputs)
                outputs = sample_ddim(self.model, x_noisy_init, inputs, timesteps=50, device=self.device)
                
                if outputs.shape[1] > 1:
                    vessel_probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    vessel_probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                
                if len(vessel_probs.shape) == 3:
                    vessel_probs = np.expand_dims(vessel_probs, axis=1)
                
                predictions.append(vessel_probs)
        
        pred_patches = np.concatenate(predictions, axis=0)
        if len(pred_patches.shape) == 3:
            pred_patches = np.expand_dims(pred_patches, axis=1)
        
        return pred_patches
    
    def evaluate_predictions(self, predictions, labels, masks=None):
        """Evaluate predictions"""
        if masks is not None:
            valid_mask = masks.flatten() > 0
            flat_preds = predictions.flatten()[valid_mask]
            flat_labels = labels.flatten()[valid_mask]
        else:
            flat_preds = predictions.flatten()
            flat_labels = labels.flatten()
        
        fpr, tpr, _ = roc_curve(flat_labels, flat_preds)
        auroc = auc(fpr, tpr)
        thresholds_to_try = np.linspace(0.1, 0.9, 9)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds_to_try:
            binary_preds = (flat_preds >= threshold).astype(np.uint8)
            f1 = f1_score(flat_labels, binary_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        evaluator = Evaluate()
        evaluator.add_batch(flat_labels, flat_preds)
        _, accuracy, specificity, sensitivity, precision_val = evaluator.confusion_matrix()
        
        metrics = {
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'precision': precision_val,
            'f1_score': evaluator.f1_score(),
            'auroc': evaluator.auc_roc(),
            'best_threshold': best_threshold
        }
        
        with open(join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Best F1: {best_f1:.4f} (threshold = {best_threshold:.2f})")
        print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}, Precision: {precision_val:.4f}, AUROC: {auroc:.4f}")
        
        return metrics, best_threshold
    
    def visualize_results(self, original_imgs, ground_truths, predictions, masks=None, num_display=3):
        num_display = min(num_display, len(original_imgs))
        vis_dir = join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        
        for i in range(num_display):
            orig_img = (original_imgs[i].squeeze() * 255).astype(np.uint8)
            gt_img = (ground_truths[i].squeeze() * 255).astype(np.uint8)
            pred_img = (predictions[i].squeeze() * 255).astype(np.uint8)
            binary_pred = ((predictions[i].squeeze() > 0.5).astype(np.uint8) * 255)
            
            cv2.imwrite(join(vis_dir, f'original_{i}.png'), orig_img)
            cv2.imwrite(join(vis_dir, f'ground_truth_{i}.png'), gt_img)
            cv2.imwrite(join(vis_dir, f'prediction_prob_{i}.png'), pred_img)
            cv2.imwrite(join(vis_dir, f'prediction_binary_{i}.png'), binary_pred)
    
    def save_predictions(self, predictions, original_shape, threshold=0.5):
        pred_dir = join(self.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, pred in enumerate(predictions):
            prob_img = (pred * 255).astype(np.uint8)
            cv2.imwrite(join(pred_dir, f'prob_map_{i}.png'), prob_img)
            
            for t in thresholds:
                binary = (pred >= t).astype(np.uint8) * 255
                cv2.imwrite(join(pred_dir, f'binary_seg_{i}_threshold_{t:.3f}.png'), binary)
    
    def test_model(self, test_data_path_list=None):
        if test_data_path_list is None:
            test_data_path_list = self.args.test_data_path_list
        
        if not os.path.exists(test_data_path_list):
            return None
        
        if not self.load_model(self.args.model_path):
            return None
        patch_height = self.config.get('test_patch_height', 96)
        patch_width = self.config.get('test_patch_width', 96)
        stride_height = self.config.get('stride_height', 16)
        stride_width = self.config.get('stride_width', 16)
        
        patches_imgs_test, test_imgs_original, test_masks, test_FOVs, new_height, new_width = get_data_test_overlap(
            test_data_path_list=test_data_path_list,
            patch_height=patch_height,
            patch_width=patch_width,
            stride_height=stride_height,
            stride_width=stride_width
        )
        
        
        test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(patches_imgs_test).float(),
            torch.zeros(patches_imgs_test.shape[0])
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers
        )
        
        pred_patches = self.generate_predictions(test_loader)
        pred_imgs = recompone_overlap(
            pred_patches, 
            new_height, 
            new_width, 
            stride_height, 
            stride_width
        )
        
        orig_height = test_imgs_original.shape[2]
        orig_width = test_imgs_original.shape[3]
        pred_imgs = pred_imgs[:, :, 0:orig_height, 0:orig_width]
        
        metrics, best_threshold = self.evaluate_predictions(
            pred_imgs, 
            test_masks, 
            test_FOVs if self.args.use_mask else None
        )
        
        self.visualize_results(
            test_imgs_original, 
            test_masks, 
            pred_imgs, 
            test_FOVs if self.args.use_mask else None,
            num_display=min(self.args.num_display, test_imgs_original.shape[0])
        )
        
        self.save_predictions(pred_imgs, test_imgs_original.shape, threshold=best_threshold)
        return metrics


def parse_test_args():
    """Parse test arguments, extending config.py arguments"""
    import argparse
    
    args = parse_args()
    parser = argparse.ArgumentParser(description='Diffusion model testing')
    parser.add_argument('--output_dir', type=str, default='diffusion/DRIVE_2_test', help='Output directory')
    parser.add_argument('--config_file', type=str, default=None, help='Config file path')
    parser.add_argument('--use_mask', action='store_true', help='Use mask for evaluation')
    parser.add_argument('--num_display', type=int, default=3, help='Number of samples to visualize')
    
    test_args, _ = parser.parse_known_args()
    
    for key, value in vars(test_args).items():
        setattr(args, key, value)
    
    if not hasattr(args, 'output_dir') or args.output_dir is None:
        args.output_dir = 'diffusion/DRIVE_2_test'
    if not hasattr(args, 'model_path') or args.model_path is None:
        args.model_path = join(args.output_dir, 'best_model.pth')
    
    return args


def main():
    setup_seed(2023)
    
    args = parse_test_args()
    
    tester = DiffusionTester(args)
    results = tester.test_model()
    
    if results is not None:
        print("\nFinal evaluation metrics:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main() 