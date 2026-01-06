import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from video_diffusion_pytorch import VAE, FrameConditioningEncoder

# ============== Configuration ==============
config = {
    # Model architecture
    'dim': 64,
    'dim_mults': (1, 2, 4, 8),
    'attn_heads': 8,
    'attn_dim_head': 32,
    
    # VAE settings
    'latent_dim': 32,  # Latent channels (compressed representation)
    'vae_base_dim': 512,
    
    # Frame conditioning
    'frame_cond_dim': 512,
    'num_prev_frames': 4,  # Number of previous frames to condition on
    
    # Video settings
    'image_size': 64,
    'num_frames': 10,  # Frames per chunk
    'channels': 3,  # RGB
    
    # Diffusion settings
    'timesteps': 1000,
    'loss_type': 'l1',
    
    # Training settings
    'train_batch_size': 1,
    'train_lr': 1e-4,
    'vae_lr': 1e-4,
    'train_num_steps': 700000,
    'gradient_accumulate_every': 2,
    'ema_decay': 0.995,
    'amp': True,
    
    # Classifier-free guidance
    'null_frame_cond_prob': 0.7,  # 10% chance to drop frame conditioning during training
    'frame_cond_scale': 1.5,  # Guidance scale for frame conditioning at inference
    
    # Checkpointing
    'save_and_sample_every': 5000,
    'results_folder': './results',
    'data_folder': './video_diffusion_pytorch/data',  # Folder containing .gif files
}

# ============== Initialize Components ==============

print("Initializing VAE...")
vae = VAE(
    channels=config['channels'],
    latent_dim=config['latent_dim'],
    base_dim=config['vae_base_dim']
)

print("Initializing Frame Conditioning Encoder...")
frame_cond_encoder = FrameConditioningEncoder(
    latent_dim=config['latent_dim'],
    feature_dim=config['frame_cond_dim'],
    num_prev_frames=config['num_prev_frames']
)

print("Initializing UNet3D...")
model = Unet3D(
    dim=config['dim'],
    dim_mults=config['dim_mults'],
    channels=config['latent_dim'],  # IMPORTANT: Use latent_dim, not image channels!
    attn_heads=config['attn_heads'],
    attn_dim_head=config['attn_dim_head'],
    frame_cond_dim=config['frame_cond_dim'],  # Enable frame conditioning
    use_sparse_linear_attn=True,
)

diffusion = GaussianDiffusion(
    model,
    vae=vae,
    frame_cond_encoder=frame_cond_encoder,
    image_size=config['image_size'],
    num_frames=config['num_frames'],
    channels=config['channels'],  # Original image channels
    timesteps=config['timesteps'],
    loss_type=config['loss_type']
    # REMOVED: num_prev_frames - not needed here
).cuda()
print("Initializing Trainer...")
trainer = Trainer(
    diffusion,
    config['data_folder'],
    train_batch_size=config['train_batch_size'],
    train_lr=config['train_lr'],
    vae_lr=config['vae_lr'],
    save_and_sample_every=config['save_and_sample_every'],
    train_num_steps=config['train_num_steps'],
    gradient_accumulate_every=config['gradient_accumulate_every'],
    ema_decay=config['ema_decay'],
    amp=config['amp'],
    num_frames=config['num_frames'],
    num_prev_frames=config['num_prev_frames'],
    results_folder=config['results_folder'],
    null_frame_cond_prob=config['null_frame_cond_prob'],
    frame_cond_scale=config['frame_cond_scale'],
    max_grad_norm=1.0  # Gradient clipping for stability
)

# ============== Start Training ==============
print(f"\nStarting training for {config['train_num_steps']} steps...")
print(f"Data folder: {config['data_folder']}")
print(f"Results folder: {config['results_folder']}")
print(f"Batch size: {config['train_batch_size']}")
print(f"Learning rate: {config['train_lr']}")
print(f"Frame conditioning: {config['num_prev_frames']} previous frames")
print(f"Null frame cond prob: {config['null_frame_cond_prob']}")
print("-" * 60)
trainer.load(-1)
trainer.train()

print("\n" + "="*60)
print("Training completed!")
print("="*60)


# ============== Inference Example ==============
"""
After training, you can generate videos like this:

# Load trained model
milestone = -1  # -1 loads the latest checkpoint
trainer.load(milestone)

# Generate a short video (conditioned on previous frames)
sample_data = next(iter(trainer.dl))
prev_frames = sample_data['prev'][:1].cuda()  # Take first sample's previous frames

video = trainer.ema_model.sample(
    batch_size=1,
    prev_frames=prev_frames,
    cond_scale=1.0,  # Text conditioning scale (if using text)
    frame_cond_scale=1.5  # Previous frame conditioning scale
)

# Generate a long video autoregressively
long_video = trainer.ema_model.sample_autoregressive(
    num_total_frames=60,  # Generate 60 frames total
    batch_size=1,
    frame_cond_scale=1.5
)

# Save the video
from video_diffusion_pytorch import video_tensor_to_gif
video_tensor_to_gif(long_video[0], 'generated_video.gif')
"""