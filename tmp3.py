import torch
from pathlib import Path

# Import your modules
from video_diffusion_guidance import (
    Unet3D,
    GaussianDiffusion,
    Trainer,
    Dataset,
    temporal_consistency_guidance,
    temporal_consistency_injection,

    VAE
)
from PIL import Image
import numpy as np
import random
def main():
    # ============== CONFIGURATION ==============
    config = {
        # diffusion Architecture
        'dim': 128,
        'dim_mults': (1, 2, 4, 8),
        'channels': 3,
        'attn_heads': 8,
        'attn_dim_head': 32,
        
        # VAE Settings
        'vae_latent_dim': 32,
        'vae_base_dim': 128,
        
        # Video Settings
        'image_size': 64,
        'num_frames': 10,
        
        # Diffusion Settings
        'timesteps': 1000,
        'loss_type': 'l1',
        'latent_scale': 1.0,
        
        # Training Settings
        'train_batch_size': 4,
        'train_lr': 1e-4,
        'train_num_steps': 400000,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'amp': True,
        'max_grad_norm': 1.0,
        
        # Checkpointing
        'save_and_sample_every': 5000,
        'num_sample_rows': 2,
        'results_folder': './results',
        'data_folder': './video_diffusion_pytorch/data',
        
        # Resume
        'resume_from': 42,  # Set to milestone number to resume
    }
    
    # ============== INITIALIZE VAE ==============
    vae = VAE(
        channels=config['channels'],
        latent_dim=config['vae_latent_dim'],
        base_dim=config['vae_base_dim']
    ).cuda()
    
    # ============== INITIALIZE UNET ==============
    unet = Unet3D(
        dim=config['dim'],
        channels=config['vae_latent_dim'],  # Operates in latent space
        out_dim=config['vae_latent_dim'],
        dim_mults=config['dim_mults'],
        attn_heads=config['attn_heads'],
        attn_dim_head=config['attn_dim_head']
    ).cuda()
    
    # ============== INITIALIZE DIFFUSION diffusion ==============
    diffusion = GaussianDiffusion(
        denoise_fn=unet,
        vae=vae,
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        channels=config['channels'],
        timesteps=config['timesteps'],
        loss_type=config['loss_type'],
        latent_scale=config['latent_scale'],
        injection_steps=600
    ).cuda()
    
    # ============== INITIALIZE TRAINER ==============
    trainer = Trainer(
        diffusion_model=diffusion,
        folder=config['data_folder'],
        ema_decay=config['ema_decay'],
        num_frames=config['num_frames'],
        train_batch_size=config['train_batch_size'],
        train_lr=config['train_lr'],
        train_num_steps=config['train_num_steps'],
        gradient_accumulate_every=config['gradient_accumulate_every'],
        amp=config['amp'],
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=config['save_and_sample_every'],
        results_folder=config['results_folder'],
        num_sample_rows=config['num_sample_rows'],
        max_grad_norm=config['max_grad_norm']
    )
    trainer.load(milestone=79)
    # trainer.train()
    def count_parameters(model):
        """Count the number of trainable parameters in a model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_sizes(vae, unet):
        """Get parameter counts for VAE components and UNet"""
        
        # Total VAE parameters
        print(vae)
        vae_total = count_parameters(vae)
        # VAE Encoder parameters
        vae_encoder = count_parameters(vae.encoder)
        
        # VAE Decoder parameters
        vae_decoder = count_parameters(vae.decoder)
        
        # UNet parameters
        unet_total = count_parameters(unet)
        
        # Print results
        print("=" * 60)
        print("MODEL PARAMETER COUNTS")
        print("=" * 60)
        print(f"VAE Encoder:        {vae_encoder:,} parameters ({vae_encoder/1e6:.2f}M)")
        print(f"VAE Decoder:        {vae_decoder:,} parameters ({vae_decoder/1e6:.2f}M)")
        print(f"VAE Total:          {vae_total:,} parameters ({vae_total/1e6:.2f}M)")
        print("-" * 60)
        print(f"UNet:               {unet_total:,} parameters ({unet_total/1e6:.2f}M)")
        print("=" * 60)
        print(f"TOTAL (VAE + UNet): {vae_total + unet_total:,} parameters ({(vae_total + unet_total)/1e6:.2f}M)")
        print("=" * 60)
        
        return {
            'vae_encoder': vae_encoder,
            'vae_decoder': vae_decoder,
            'vae_total': vae_total,
            'unet': unet_total,
            'total': vae_total + unet_total
        }

    # Usage after initializing your models:
    param_counts = get_model_sizes(vae, unet)

        # ============== START TRAINING ==============
    def save_video_as_gif(frames, save_path, fps=100, loop=0):
        """
        Save video frames as a GIF file
        
        Args:
            frames: Tensor of shape (B, C, F, H, W) in range [0, 1]
                    or (C, F, H, W) for single video
            save_path: Path to save the GIF (e.g., 'output.gif')
            fps: Frames per second
            loop: Number of loops (0 = infinite loop)
        """
        # Handle batch dimension
        if frames.ndim == 5:
            frames = frames[0]  # Take first video in batch
        
        # Convert to numpy: (C, F, H, W) -> (F, H, W, C)
        frames = frames.cpu().numpy()
        frames = np.transpose(frames, (1, 2, 3, 0))
        
        # Convert to uint8 [0, 255]
        frames = (frames * 255).astype(np.uint8)
        
        # Handle grayscale
        if frames.shape[-1] == 1:
            frames = frames.squeeze(-1)
        
        # Convert each frame to PIL Image
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        duration = int(1000 / fps)  # milliseconds per frame
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=loop
        )
        print(f"Saved GIF to {save_path}")


    # Usage:
    # Generate frames


    for j in range(10):
        a = random.randint(0, 10000)
        frames, latents = diffusion.sample(batch_size=1)
        # For long video with multiple chunks:
        all_frames_inject = []
        all_frames_guidance = []
        all_frames_inject.append(frames)
        all_frames_guidance.append(frames)
        latents_inject=latents
        prev_frames=frames[:, :, -5:].detach().clone()
        
        for i in range(0):  # Generate 5 chunks
            injection_fn = temporal_consistency_injection(diffusion,latents_inject, 5)
            frames_inject, latents_inject = diffusion.sample(batch_size=1,  injection_fn=injection_fn)
            frames_inject = frames_inject[:, :, 5:] 
            all_frames_inject.append(frames_inject)
        long_video = torch.cat(all_frames_inject, dim=2)  # (B, C, total_F, H, W)
        save_video_as_gif(long_video, f'long_video_{a}_injection_{diffusion.injection_steps}.gif', fps=1)
        for i in range(3):  # Generate 5 chunks
            guidance_fn = temporal_consistency_guidance(diffusion, prev_frames, 5, 30)
            frames_guidance, latents_guidance = diffusion.sample(batch_size=1,  guidance_fn=guidance_fn)
            prev_frames = frames_guidance[:, :, -5:].detach().clone()
            frames_guidance = frames_guidance[:, :, 5:] 
            all_frames_guidance.append(frames_guidance)
        long_video = torch.cat(all_frames_guidance, dim=2)  # (B, C, total_F, H, W)
        save_video_as_gif(long_video, f'long_video_{a}_guidance.gif', fps=1)


if __name__ == '__main__':
    main()