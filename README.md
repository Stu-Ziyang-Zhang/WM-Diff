# WM-Diff: Wavelet-Mamba Diffusion Model for Accurate Retinal Vessel Segmentation

## Overview

WM-Diff is a deep learning framework for retinal vessel segmentation that integrates wavelet transforms, Mamba state space models, and diffusion probabilistic models. The framework employs a multi-scale conditional guidance network to inject hierarchical structural priors into the diffusion process, leveraging powerful image reconstruction capabilities to enhance segmentation accuracy.

## Architecture

The WM-Diff framework comprises three core components:

### 1. Diffusion Module
- **DiffusionCondUNet** : Conditional diffusion UNet implementing a noise-to-image generation process using DDPM with 100 timesteps, progressively refining vessel structures through iterative denoising.
- **TripletAttention**  Triple-channel parallel architecture capturing feature dependencies across channel-width (C-W), height-channel (H-C), and height-width (H-W) dimensions, incorporating SS2D state space models and attention gating mechanisms.

### 2. Condition Module
- **DynamicWaveletBlock** : Differentiable multi-basis wavelet module (DMBWM) fusing Haar and Daubechies-2 (db2) wavelets for edge detection and texture discrimination.
- **VascularWaveletDown** : Multi-basis wavelet downsampling (MBWD) preserving multi-scale spatial details via forward discrete wavelet transform.
- **VascularWaveletUp** : Multi-basis wavelet upsampling (MBWU) reconstructing high-quality features via inverse discrete wavelet transform.
- **CrossAttention** : Conditional feature selector enabling adaptive feature interaction through cross-attention mechanisms.

### 3. VDDA (Vessel Dynamic Deformable Attention)
- **DeformableAttention** : Deformable attention mechanism with learnable offset networks for dynamic sampling position prediction, reducing computational complexity from O(C²) to O(C²/G).
- **EfficientMultiScaleTransformer** : Multi-scale transformer with adaptive receptive field adjustment based on vessel calibers.
- **HybridFusionModule** : Hybrid fusion module integrating multi-scale features via LiteSKTransformer and EfficientMultiScaleTransformer.
