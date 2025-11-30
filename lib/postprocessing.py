import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

def refine_boundaries(pred_img, threshold=0.5):
    if pred_img is None:
        print("Error: Input prediction map is empty")
        return np.zeros((10, 10))
    
    if len(pred_img.shape) > 2:
        pred_img = pred_img.squeeze()
        
    if not np.isfinite(pred_img).all():
        print("Warning: Prediction map contains invalid values (NaN/Inf), attempting to fix")
        pred_img = np.nan_to_num(pred_img, nan=0.0, posinf=1.0, neginf=0.0)
        
    pred_img = np.clip(pred_img, 0, 1)
    
    binary_mask = (pred_img > threshold).astype(np.uint8)
    
    kernel_size = max(3, min(pred_img.shape) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    boundary_region = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel) > 0
    
    dilated_boundary = binary_dilation(boundary_region, iterations=2)
    
    local_mean = gaussian_filter(pred_img, sigma=kernel_size/2)
    local_std = np.sqrt(gaussian_filter(pred_img**2, sigma=kernel_size/2) - local_mean**2)
    
    adaptive_threshold = local_mean + 0.5 * local_std
    adaptive_threshold = np.clip(adaptive_threshold, 0.3, 0.7)
    
    enhanced_pred = pred_img.copy()
    enhanced_pred[dilated_boundary] = np.where(
        pred_img[dilated_boundary] > adaptive_threshold[dilated_boundary],
        np.clip(pred_img[dilated_boundary] * 1.2, 0, 1),
        np.clip(pred_img[dilated_boundary] * 0.8, 0, 1)
    )
    
    binary_enhanced = (enhanced_pred > threshold).astype(np.uint8)
    
    try:
        if len(binary_enhanced.shape) > 2:
            binary_enhanced_2d = binary_enhanced.squeeze()
        else:
            binary_enhanced_2d = binary_enhanced
            
        if binary_enhanced_2d.dtype != np.uint8:
            binary_enhanced_2d = binary_enhanced_2d.astype(np.uint8)
            
        if np.sum(binary_enhanced_2d) == 0:
            print("Warning: Binary image is empty or all black, skipping connected component analysis")
        else:
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_enhanced_2d)
            
            if retval > 1:
                for i in range(1, retval):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < 10:
                        binary_enhanced_2d[labels == i] = 0
                        
                if len(binary_enhanced.shape) > 2:
                    binary_enhanced = np.expand_dims(binary_enhanced_2d, axis=0)
                else:
                    binary_enhanced = binary_enhanced_2d
    except Exception as e:
        print(f"Warning: Connected component analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    final_pred = pred_img.copy()
    final_pred[dilated_boundary] = enhanced_pred[dilated_boundary]
    
    return final_pred


def process_segmentation_result(pred_img, orig_img=None, threshold=0.5):
    if len(pred_img.shape) > 2:
        pred_img = pred_img.squeeze()
    
    refined_pred = refine_boundaries(pred_img, threshold)
    
    if orig_img is not None:
        try:
            if len(orig_img.shape) > 2:
                orig_img = orig_img.squeeze()
            
            if not np.isfinite(orig_img).all():
                print("Warning: Original image contains invalid values, skipping edge alignment")
            else:
                norm_img = np.clip(orig_img, 0, 1) * 255
                edges = cv2.Canny(norm_img.astype(np.uint8), threshold1=30, threshold2=100)
                edge_mask = edges > 0
                dilated_edge = cv2.dilate(edge_mask.astype(np.uint8), None, iterations=2) > 0
                refined_pred[dilated_edge] = np.where(
                    refined_pred[dilated_edge] > threshold,
                    np.clip(refined_pred[dilated_edge] * 1.1, 0, 1),
                    refined_pred[dilated_edge] * 0.9
                )
        except Exception as e:
            print(f"Warning: Edge alignment processing failed: {e}")
    binary_pred = (refined_pred > threshold).astype(np.uint8)
    
    return refined_pred, binary_pred

