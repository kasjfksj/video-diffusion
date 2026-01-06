import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
from einops import rearrange
from pathlib import Path
from PIL import Image
from torchvision import transforms as TF

from functools import partial
import sys

# Enable optimizations
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Additional CUDA optimizations
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Hyperparameters
NUM_FRAMES = 16
IMAGE_SIZE = 64
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
TRAIN_STEPS = 40000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COND_DIM = 2

# ODE solver settings
NUM_SAMPLING_STEPS = 10

# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# Sinusoidal position embedding for time t ∈ [0, 1]
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Pre-compute log factor
        self.register_buffer('log_factor', torch.tensor(math.log(10000) / (dim // 2 - 1)))

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -self.log_factor)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm2D(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # Use fused GroupNorm for better performance
        return F.group_norm(x, 1, self.gamma.squeeze(), None, self.eps)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_c * 2)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            LayerNorm2D(in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            LayerNorm2D(out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            B = x.shape[0]
            if time_emb.shape[0] != B:
                time_emb = time_emb[:B]
            
            cond = self.mlp(time_emb)
            scale, shift = cond.chunk(2, dim=1)

            # More efficient broadcasting
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = LayerNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (head d) x y -> b head (x y) d', head=self.heads), qkv)
        
        # Use scaled_dot_product_attention for optimized attention
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            q = q * self.scale
            sim = torch.einsum('bhid,bhjd->bhij', q, k)
            attn = sim.softmax(dim=-1)
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        out = rearrange(out, 'b head (x y) d -> b (head d) x y', head=self.heads, x=h, y=w)
        out = self.to_out(out)
        
        return out + x


class VelocityNet(nn.Module):
    def __init__(self, init_dim=32, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        dims = [init_dim] + [init_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        time_dim = init_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        motion_dim = 1
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.SiLU()
        )
        
        combined_dim = time_dim + motion_dim
        
        self.init_conv = nn.Conv2d(3, init_dim, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock2D(dim_in, dim_out, combined_dim),
                ResnetBlock2D(dim_out, dim_out, combined_dim),
                SpatialAttention(dim_out, heads=4),
                nn.Conv2d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock2D(mid_dim, mid_dim, combined_dim)
        self.mid_attn = SpatialAttention(mid_dim, heads=4)
        self.mid_block2 = ResnetBlock2D(mid_dim, mid_dim, combined_dim)
        
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock2D(dim_out * 2, dim_in, combined_dim),
                ResnetBlock2D(dim_in, dim_in, combined_dim),
                SpatialAttention(dim_in, heads=4),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        self.final_conv = nn.Sequential(
            LayerNorm2D(init_dim),
            nn.SiLU(),
            nn.Conv2d(init_dim, 3, 1)
        )

    def forward(self, x, t, cond):
        batch_size = x.shape[0]
        
        # Convert x from [B, H, W, 3] to [B, 3, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Ensure t is [B]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Convert cond from [B] to [B, 1]
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)
        elif cond.dim() == 0:
            cond = cond.unsqueeze(0).unsqueeze(-1)
        
        # Embeddings
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        emb = torch.cat([t_emb, c_emb], dim=-1)
        
        x = self.init_conv(x)
        
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            x = upsample(x)
        
        velocity = self.final_conv(x)
        velocity = velocity.permute(0, 2, 3, 1).contiguous()
        
        return velocity


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 64
CHANNELS = 3
LATENT_DIM = 3
NUM_SAMPLING_STEPS = 50
NUM_FRAMES = 16

import torchvision.utils as vutils
class FlowMatching(nn.Module):
    def __init__(self, velocity_net, sigma_max: float = 0.5):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_max = sigma_max

    def _quadratic_noise(self, tau: torch.Tensor) -> torch.Tensor:
        return self.sigma_max * tau * (1.0 - tau)

    def get_trajectory(
        self,
        x_prev: torch.Tensor,
        x_cur: torch.Tensor,
        tau: torch.Tensor,
        add_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tau = tau.view(-1, 1, 1, 1)
        x_t = x_prev + tau * (x_cur - x_prev)  # Fused operation

        epsilon = None
        if add_noise and self.sigma_max > 0.0:
            epsilon = torch.randn_like(x_prev)
            sigma_t = self._quadratic_noise(tau.squeeze(1)).view(-1, 1, 1, 1)
            x_t = x_t + sigma_t * epsilon
        return x_t, epsilon

    def get_target_velocity(
        self,
        x_prev: torch.Tensor,
        x_cur: torch.Tensor,
        tau: torch.Tensor,
        epsilon: torch.Tensor | None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        v = x_cur - x_prev
        if add_noise and epsilon is not None:
            tau = tau.view(-1, 1, 1, 1)
            d_sigma_dtau = self.sigma_max * (1.0 - 2.0 * tau)
            v = v + d_sigma_dtau * epsilon
        return v

    def forward(
        self,
        pair: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B = pair.shape[0]
        device = pair.device

        x_prev = pair[:, 0]
        x_cur = pair[:, 1]

        tau = torch.rand(B, device=device)

        x_t, epsilon = self.get_trajectory(x_prev, x_cur, tau, add_noise=False)
        v_target = self.get_target_velocity(x_prev, x_cur, tau, epsilon, add_noise=False)
        
        v_pred = self.velocity_net(x_t, tau, cond)
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        prev_frame: torch.Tensor,  # The previous frame to start from
        cond: torch.Tensor,
        num_steps: int = NUM_SAMPLING_STEPS,
        method: str = "euler",
    ) -> torch.Tensor:
        batch_size = prev_frame.shape[0]
        
        # Start from previous frame (τ=0), not random noise!
        x = prev_frame.clone()
        
        # Ensure x is in the correct range [-1, 1]
        if x.max() > 1.0 or x.min() < -1.0:
            print(f"Warning: prev_frame range is [{x.min():.3f}, {x.max():.3f}], expected [-1, 1]")
        
        dt = 1.0 / num_steps
        tau_vec = torch.linspace(0.0, 1.0, num_steps + 1, device=DEVICE)
        
        for i in range(num_steps):
            tau = tau_vec[i].expand(batch_size)
            
           
            
            v = self.velocity_net(x, tau, cond)
            
            
            
            if method == "euler":
                x = x + dt * v
            elif method == "heun":
                x_mid = x + dt * v
                tau_next = tau_vec[i + 1].expand(batch_size)
                v_next = self.velocity_net(x_mid, tau_next, cond)
                x = x + dt * 0.5 * (v + v_next)
            
            # Optionally clamp during generation to prevent explosion
            # x = x.clamp(-2, 2)  # Allow some overflow but prevent extreme values
        
        print(f"Final x range before conversion: [{x.min():.3f}, {x.max():.3f}]")
        
        # x is now the predicted next frame at τ=1
        # Check the expected output format
        if x.shape[1] == CHANNELS or len(x.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
            x = x.clamp(-1, 1)
            x = (x + 1) / 2
            x = (x * 255).round().clamp(0, 255).to(torch.uint8)
        
        return x




class ConditionalVideoDataset(Dataset):
    def __init__(self, video_path, image_size=IMAGE_SIZE, 
                 single_clip_idx=None, camera_motion=None):
        data = np.load(video_path)

        # Pre-process video to tensors
        video = torch.from_numpy((data['video'].astype(np.float32) / 255.0) * 2 - 1)

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        cur_h, cur_w = video.shape[1], video.shape[2]
        if (cur_h, cur_w) != self.image_size:
            print(f"Resizing video from {(cur_h, cur_w)} to {self.image_size}")
            vid_3c = video.permute(0, 3, 1, 2)
            vid_3c = torch.nn.functional.interpolate(
                vid_3c, size=self.image_size, mode='bilinear', align_corners=False)
            self.video = vid_3c.permute(0, 2, 3, 1)
        else:
            self.video = video

        # Process motion
        if camera_motion is not None:
            if isinstance(camera_motion, list):
                self.camera_motion = camera_motion[0] if camera_motion else None
            else:
                self.camera_motion = camera_motion
            if isinstance(self.camera_motion, np.ndarray):
                self.camera_motion = torch.from_numpy(self.camera_motion).float()
        elif 'actions' in data:
            self.camera_motion = torch.from_numpy(data['actions']).float()
        else:
            print("Warning: no motion, using zeros")
            self.camera_motion = torch.zeros(self.video.shape[0])

        if self.camera_motion.dim() > 1:
            self.camera_motion = self.camera_motion.mean(dim=1)

        self.single_clip_idx = single_clip_idx
        self.max_start = self.video.shape[0] - 1

        if single_clip_idx is not None:
            assert 0 <= single_clip_idx < self.max_start
            start = single_clip_idx
            self.single_clip = self.video[start:start+2]
            self.single_motion = self.camera_motion[start].item()
            self.length = 10000
        else:
            self.length = self.max_start

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.single_clip_idx is not None:
            return self.single_clip.clone(), self.single_motion

        idx = idx % self.max_start
        start = idx

        clip = self.video[start:start+2]
        motion = self.camera_motion[start]

        return clip, motion


def train():
    print(f"Training Flow Matching on device: {DEVICE}")
    
    folder = '0.npz'
    SINGLE_CLIP_INDEX = 0
    dataset = ConditionalVideoDataset(folder, image_size=IMAGE_SIZE)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Increased from 2
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  # Prefetch batches
    )
    with open('camera_motion.txt', 'w') as f:
        for i, val in enumerate(dataset.camera_motion):
            f.write(f"{val.item()}\n")
    print("Saved camera_motion.txt")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Training mode: {'SINGLE CLIP (overfitting)' if hasattr(dataset, 'single_clip') else 'ALL CLIPS (normal)'}")
    
    velocity_net = VelocityNet().to(DEVICE)
    flow_model = FlowMatching(velocity_net).to(DEVICE)
    
    # Compile model for PyTorch 2.0+ (skip on older GPUs with CUDA capability < 7.0)
    if hasattr(torch, 'compile'):
        try:
            # Check CUDA capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 7:  # Only compile on GPUs with CUDA capability >= 7.0
                    print(f"Compiling model (CUDA capability: {capability[0]}.{capability[1]})")
                    flow_model = torch.compile(flow_model, mode='reduce-overhead')
                else:
                    print(f"Skipping torch.compile (CUDA capability {capability[0]}.{capability[1]} < 7.0)")
            else:
                flow_model = torch.compile(flow_model, mode='reduce-overhead')
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")
    
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=LEARNING_RATE, 
                                   weight_decay=0.01, fused=True if DEVICE == 'cuda' else False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    step = 0
    pbar = tqdm(total=TRAIN_STEPS, desc='Training Flow Matching')
    
    while step < TRAIN_STEPS:
        for video_clip, camera_motion in dataloader:
            if step >= TRAIN_STEPS:
                break
            
            video_clip = video_clip.to(DEVICE, non_blocking=True)
            camera_motion = camera_motion.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = flow_model(video_clip, camera_motion)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = flow_model(video_clip, camera_motion)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            if step % 2500 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                print(f'\nStep {step}, Loss: {loss.item():.4f}')
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    torch.save({
        'model': velocity_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, 'flow_matching_video_final.pth')
    
    print('Training complete!')

if __name__ == '__main__':
    train()