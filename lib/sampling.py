import torch
from diffusion_framework import T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, get_index_from_list

def sample_ddim(model, x_noisy, x_original, timesteps, device='cuda', eta=0.0):
    model.eval()
    x = x_noisy.clone()
    
    t_list = torch.linspace(T-1, 0, timesteps+1, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i in range(timesteps):
            t = t_list[i].unsqueeze(0).repeat(x.shape[0])
            t_next = t_list[i+1].unsqueeze(0).repeat(x.shape[0]) if i < timesteps-1 else torch.zeros(x.shape[0], dtype=torch.long, device=device)
            
            predicted_noise = model(x, t, x_original=x_original, train_cond_net=False)
            
            if predicted_noise.shape[1] > 1:
                predicted_noise = predicted_noise[:, 0:1]
            
            sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
            
            pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / (sqrt_alphas_cumprod_t + 1e-8)
            pred_x0 = torch.clamp(pred_x0, 0, 1)
            
            if i < timesteps - 1:
                sqrt_alphas_cumprod_t_next = get_index_from_list(sqrt_alphas_cumprod, t_next, x.shape)
                sqrt_one_minus_alphas_cumprod_t_next = get_index_from_list(sqrt_one_minus_alphas_cumprod, t_next, x.shape)
                
                pred_dir = sqrt_one_minus_alphas_cumprod_t_next * predicted_noise
                x = sqrt_alphas_cumprod_t_next * pred_x0 + pred_dir
            else:
                x = pred_x0
    
    return x

