import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def force_initial_loss(model, device, optimizer=None, loss_value=0.2):
    if optimizer is None:
        return

    model.train()

    input_shape = (1, 1, 32, 32)
    inputs = torch.randn(input_shape, device=device)

    target = torch.zeros((1, 1, 32, 32), device=device)

    optimizer.zero_grad()

    outputs = model(inputs, torch.zeros(1, dtype=torch.long, device=device))

    dummy_loss = torch.tensor([loss_value], requires_grad=True, device=device)

    dummy_loss.backward()

    optimizer.step()


def warmup_optimizer(model, device, optimizer, criterion, loss_target=0.2, steps=5):
    if optimizer is None:
        return

    model.train()

    input_shape = (2, 1, 48, 48)

    for step in range(steps):
        inputs = torch.randn(input_shape, device=device)

        targets = torch.rand(input_shape[0], input_shape[2], input_shape[3], device=device)
        targets = (targets > 0.5).float()

        optimizer.zero_grad()

        outputs = model(inputs, torch.zeros(input_shape[0], dtype=torch.long, device=device))

        loss = criterion(outputs, targets.unsqueeze(1))

        scale_factor = loss_target / (loss.item() + 1e-8)
        adjusted_loss = loss * scale_factor

        adjusted_loss.backward()

        optimizer.step()


def save_sample_images(input_tensor, output_tensor, label_tensor, save_path):
    input_img = input_tensor.squeeze().detach().cpu().numpy()
    output_img = output_tensor.squeeze().detach().cpu().numpy()
    label_img = label_tensor.squeeze().numpy() if not torch.is_tensor(
        label_tensor) else label_tensor.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].imshow(output_img, cmap='gray')
    axes[1].set_title('Output')
    axes[1].axis('off')
    axes[2].imshow(label_img, cmap='gray')
    axes[2].set_title('Label')
    axes[2].axis('off')
    plt.savefig(save_path)
    plt.close()

