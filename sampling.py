import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer, video_tensor_to_gif
from video_diffusion_pytorch import VAE, FrameConditioningEncoder

# ============== Configuration ==============
config = {
    # Model architecture
    'dim': 64,
    'dim_mults': (1, 2, 4, 8),
    'attn_heads': 8,
    'attn_dim_head': 32,
    
    # VAE settings
    'latent_dim': 8,  # Latent channels (compressed representation)
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
    'null_frame_cond_prob': 0.1,  # 10% chance to drop frame conditioning during training
    'frame_cond_scale': 7,  # Guidance scale for frame conditioning at inference
    
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
    cond_dim=config['frame_cond_dim'],
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

print("Initializing Gaussian Diffusion...")
diffusion = GaussianDiffusion(
    model,
    vae=vae,
    frame_cond_encoder=frame_cond_encoder,
    image_size=config['image_size'],
    num_frames=config['num_frames'],
    channels=config['channels'],  # Original image channels
    timesteps=config['timesteps'],
    loss_type=config['loss_type'],
    num_prev_frames=config['num_prev_frames']
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
    cond_dim=config['frame_cond_dim'],
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

print("Initializing Gaussian Diffusion...")
diffusion = GaussianDiffusion(
    model,
    vae=vae,
    frame_cond_encoder=frame_cond_encoder,
    image_size=config['image_size'],
    num_frames=config['num_frames'],
    channels=config['channels'],  # Original image channels
    timesteps=config['timesteps'],
    loss_type=config['loss_type'],
    num_prev_frames=config['num_prev_frames']
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
# Load your trained model (assuming trainer is already set up)
trainer.load(milestone=23)  # Load latest checkpoint

# Option 3: Generate long video autoregressively
long_video = trainer.ema_model.sample_autoregressive(
    num_total_frames=70,
    batch_size=1,
    frame_cond_scale=1.5
)
video_tensor_to_gif(long_video[0], 'long_output.gif')

# Option 4: Generate multiple videos as grid
videos = trainer.ema_model.sample(batch_size=4)
from einops import rearrange
import torch.nn.functional as F

videos = F.pad(videos, (2, 2, 2, 2))
grid = rearrange(videos, '(i j) c f h w -> c f (i h) (j w)', i=2)
video_tensor_to_gif(grid, 'grid_output.gif')