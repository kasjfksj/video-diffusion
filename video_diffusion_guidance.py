import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def save_gif(prev_frames, current_frames):
    frames = torch.cat([prev_frames, current_frames], dim=2)
    frames = frames.squeeze(0).permute(1, 2, 3, 0)
    frames = (frames.clamp(0, 1) * 255).byte().cpu()
    Image.fromarray(frames[0].numpy()).save(
        "0.gif",
        save_all=True,
        append_images=[Image.fromarray(f.numpy()) for f in frames[1:]],
        duration=100,
        loop=0
    )

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class VAEEncoder(nn.Module):
    def __init__(self, channels=3, latent_dim=4, base_dim=32):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, base_dim, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv3d(base_dim, base_dim * 2, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.GroupNorm(16, base_dim * 2),
            nn.SiLU(),
            nn.Conv3d(base_dim * 2, base_dim * 4, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.GroupNorm(32, base_dim * 4),
            nn.SiLU(),
            nn.Conv3d(base_dim * 4, base_dim * 4, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(32, base_dim * 4),
            nn.SiLU(),
        )
        self.to_mean = nn.Conv3d(base_dim * 4, latent_dim, 1)
        self.to_logvar = nn.Conv3d(base_dim * 4, latent_dim, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        mean = self.to_mean(h)
        logvar = self.to_logvar(h)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=4, channels=3, base_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_dim, base_dim * 4, 1),
            nn.GroupNorm(32, base_dim * 4),
            nn.SiLU(),
            nn.Conv3d(base_dim * 4, base_dim * 4, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(32, base_dim * 4),
            nn.SiLU(),
            nn.ConvTranspose3d(base_dim * 4, base_dim * 2, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.GroupNorm(16, base_dim * 2),
            nn.SiLU(),
            nn.ConvTranspose3d(base_dim * 2, base_dim, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.ConvTranspose3d(base_dim, channels, (1, 4, 4), (1, 2, 2), (0, 1, 1)),
        )
        
    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, channels=3, latent_dim=4, base_dim=64):
        super().__init__()
        self.encoder = VAEEncoder(channels, latent_dim, base_dim)
        self.decoder = VAEDecoder(latent_dim, channels, base_dim)
        self.latent_dim = latent_dim
        self.register_buffer('latent_scale', torch.ones(1))
        self.register_buffer('latent_shift', torch.zeros(1))
        
    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode_to_latent(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        z_scaled = (z - self.latent_shift) / self.latent_scale
        return z_scaled
    
    def decode_from_latent(self, z_scaled):
        z = z_scaled * self.latent_scale + self.latent_shift
        return self.decode(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
    
    @torch.no_grad()
    def compute_latent_stats(self, dataloader, num_batches=100):
        print("Computing latent statistics for scaling...")
        self.eval()
        latents = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = batch.cuda() if not isinstance(batch, dict) else batch['current'].cuda()
            x = normalize_img(x)
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            latents.append(z)
        latents = torch.cat(latents, dim=0)
        latent_mean = latents.mean()
        latent_std = latents.std()
        print(f"Latent mean: {latent_mean.item():.4f}, std: {latent_std.item():.4f}")
        self.latent_shift.copy_(latent_mean)
        self.latent_scale.copy_(latent_std)
        self.train()
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate_queries_or_keys(self, t):
        seq_len = t.shape[-2]
        freqs = torch.einsum("i , j -> i j", torch.arange(seq_len, device=t.device).type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(t.device)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return t * cos_emb + self._rotate_half(t) * sin_emb

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)
class TemporalSpatialBlock(nn.Module):
    """Factorized 3D convolution: temporal -> spatial with proper integration"""
    def __init__(self, dim, dim_out, kernel_size=3):
        super().__init__()
        # Temporal convolution FIRST
        self.temporal_conv = nn.Conv3d(dim, dim, kernel_size=(kernel_size, 1, 1), 
                                       padding=(kernel_size//2, 0, 0), groups=dim)
        self.temporal_norm = nn.GroupNorm(8, dim)
        
        # Then spatial convolution
        self.spatial_conv = nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), 
                                      padding=(0, 1, 1))
        self.spatial_norm = nn.GroupNorm(8, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        # Temporal processing
        h = self.temporal_conv(x)
        h = self.temporal_norm(h)
        h = self.act(h)
        
        # Spatial processing
        h = self.spatial_conv(h)
        h = self.spatial_norm(h)
        h = self.act(h)
        return h
class ResnetBlock(nn.Module):
    """ResNet block with proper temporal integration"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, temporal_kernel=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        # First path: temporal -> spatial
        self.ts_block1 = TemporalSpatialBlock(dim, dim_out, kernel_size=temporal_kernel)
        
        # Second path: temporal -> spatial
        self.ts_block2 = TemporalSpatialBlock(dim_out, dim_out, kernel_size=temporal_kernel)
        
        # Residual connection that INCLUDES temporal processing
        if dim != dim_out:
            self.res_conv = nn.Sequential(
                nn.Conv3d(dim, dim_out, kernel_size=(temporal_kernel, 1, 1), 
                         padding=(temporal_kernel//2, 0, 0)),
                nn.GroupNorm(8, dim_out)
            )
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        # Store residual
        res = self.res_conv(x)
        
        # First temporal-spatial block
        h = self.ts_block1(x)
        
        # Apply time embedding
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        
        # Second temporal-spatial block
        h = self.ts_block2(h)
        
        # Add residual
        return h + res

class TemporalAttention(nn.Module):
    """Pure temporal attention that operates along frame dimension"""
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None):
        b, c, f, h, w = x.shape
        # Rearrange to treat each spatial position independently
        x = rearrange(x, 'b c f h w -> (b h w) f c')
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        q = q * self.scale
        
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        
        if exists(pos_bias):
            # pos_bias shape should be (heads, frames, frames)
            sim = sim + pos_bias
        
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return rearrange(out, '(b h w) f c -> b c f h w', b=b, h=h, w=w)


class SpatialAttention(nn.Module):
    """Pure spatial attention that operates within each frame"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        b, c, f, h, w = x.shape
        # Rearrange to treat each frame independently
        x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return rearrange(out, '(b f) (h w) c -> b c f h w', b=b, f=f, h=h, w=w)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x.shape[-2], x.device
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        if exists(pos_bias):
            sim = sim + pos_bias
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)
            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        if attn.dtype != v.dtype:
            attn = attn.to(v.dtype)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

class Unet3D(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), channels=4, 
                 attn_heads=8, attn_dim_head=32, init_dim=None, init_kernel_size=7,
                 temporal_kernel=3):
        super().__init__()
        self.channels = channels
        
        # Rotary embeddings for temporal attention
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        
        # Position bias for temporal attention
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=300)
        
        # Initial convolution
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), 
                                   padding=(0, init_padding, init_padding))
        
        # Initial temporal attention
        self.init_temporal_attn = Residual(PreNorm(init_dim, TemporalAttention(init_dim, heads=attn_heads, 
                                                                                dim_head=attn_dim_head, 
                                                                                rotary_emb=rotary_emb)))
        
        # Dimension setup
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Time embedding
        time_dim = dim * 8
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, temporal_kernel=temporal_kernel),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, temporal_kernel=temporal_kernel),
                Residual(PreNorm(dim_out, TemporalAttention(dim_out, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                Residual(PreNorm(dim_out, SpatialAttention(dim_out, heads=attn_heads, dim_head=attn_dim_head))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        
        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, temporal_kernel=temporal_kernel)
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, TemporalAttention(mid_dim, heads=attn_heads, 
                                                                              dim_head=attn_dim_head, 
                                                                              rotary_emb=rotary_emb)))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, SpatialAttention(mid_dim, heads=attn_heads, 
                                                                            dim_head=attn_dim_head)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, temporal_kernel=temporal_kernel)
        
        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, temporal_kernel=temporal_kernel),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, temporal_kernel=temporal_kernel),
                Residual(PreNorm(dim_in, TemporalAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                Residual(PreNorm(dim_in, SpatialAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        
        # Final output
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, temporal_kernel=temporal_kernel),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        # Get temporal position bias
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        
        # Initial convolution and temporal attention
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        r = x.clone()
        
        # Time embedding
        t = self.time_mlp(time)
        
        # Encoder
        h = []
        for block1, block2, temporal_attn, spatial_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = spatial_attn(x)
            h.append(x)
            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)
        
        # Decoder
        for block1, block2, temporal_attn, spatial_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = spatial_attn(x)
            x = upsample(x)
        
        # Final output
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, vae, image_size, num_frames, channels=3, timesteps=1000, loss_type='l1', use_dynamic_thres=False, dynamic_thres_percentile=0.9, latent_scale=1.0,injection_steps = 600):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.vae = vae
        self.injection_steps = injection_steps
        print(self.injection_steps)
        self.latent_scale = latent_scale
        for param in self.vae.parameters():
            param.requires_grad = True
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        

    def p_mean_variance(self, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, guidance_fn=None, guidance_scale=1.0, injection_fn=None):
        b, *_, device = *x.shape, x.device
        if injection_fn is not None:
            x = injection_fn(x, t)
        # Apply guidance to x BEFORE computing model predictions
        elif guidance_fn is not None:
            with torch.enable_grad():
                guidance_grad = guidance_fn(x, t)
                # Scale by variance (sigma_t^2)
                sigma_t_sq = extract(self.betas, t, x.shape)  # or appropriate variance term
                x = x + guidance_scale * sigma_t_sq * guidance_grad
        
        # Now compute denoising step with guided x
        with torch.no_grad():
            model_mean, _, model_log_variance = self.p_mean_variance(
                x=x, t=t, clip_denoised=clip_denoised
            )
            noise = torch.randn_like(x)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape, guidance_fn=None, guidance_scale=1.0, injection_fn=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop', total=self.num_timesteps):

            img = self.p_sample(
                img, 
                torch.full((b,), i, device=device, dtype=torch.long),
                guidance_fn=guidance_fn,
                guidance_scale=guidance_scale,
                injection_fn=injection_fn
            )

        return img

    def sample(self, batch_size=16, guidance_fn=None, guidance_scale=1.0, injection_fn=None):
        device = next(self.denoise_fn.parameters()).device
        latent_size = self.image_size // 8
        latent_channels = self.vae.latent_dim
        latents = self.p_sample_loop(
            (batch_size, latent_channels, self.num_frames, latent_size, latent_size),
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
            injection_fn=injection_fn
        )
        self.vae.eval()
        with torch.no_grad():
            samples = self.vae.decode(latents / self.latent_scale)
            samples = (samples.clamp(-1, 1) + 1) / 2
        return samples, latents

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        x_start_normalized = (x_start * 2.0 - 1.0).clamp(-1, 1)
        self.vae.eval()
        with torch.no_grad():
            x_start_latent, _ = self.vae.encode(x_start_normalized)
        x_start_latent = (x_start_latent * self.latent_scale).clamp(-10, 10)
        noise = default(noise, lambda: torch.randn_like(x_start_latent))
        x_noisy = self.q_sample(x_start=x_start_latent, t=t, noise=noise)
        predicted_noise = self.denoise_fn(x_noisy, t)
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, *args, **kwargs):
        b, device = x.shape[0], x.device
        x = x.clamp(0, 1)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
def temporal_consistency_injection(model, prev_latents, overlap_frames, save_gif=True, gif_dir="./injection_gifs"):
    """Direct injection of overlapping frames."""
    import os
    from PIL import Image
    
    if save_gif:
        os.makedirs(gif_dir, exist_ok=True)
    
    def injection_fn(z_t, t):
        if prev_latents is None:
            return z_t
        
        # Only inject during first injection_steps

        if t[0].item() < model.num_timesteps-model.injection_steps:
            return z_t
        
        # Take last frames from previous chunk
        prev_overlap = prev_latents[:, :, -overlap_frames:]
        
        # Noise to current timestep
        sqrt_alpha = extract(model.sqrt_alphas_cumprod, t, prev_overlap.shape)
        sqrt_one_minus_alpha = extract(model.sqrt_one_minus_alphas_cumprod, t, prev_overlap.shape)
        noise = torch.randn_like(prev_overlap)
        noised_overlap = sqrt_alpha * prev_overlap + sqrt_one_minus_alpha * noise
        
        # Replace first frames
        z_t[:, :, :overlap_frames] = noised_overlap
        
        # Save as GIF
        if save_gif and t[0].item() % 100 == 0:
            with torch.no_grad():
                video = model.vae.decode(z_t / model.latent_scale)
                video = (video.clamp(-1, 1) + 1) / 2  # [B, C, T, H, W] in [0, 1]
                
                # Convert to numpy and save
                video_np = video[0].cpu().permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
                video_np = (video_np * 255).astype('uint8')
                
                # Create list of PIL images
                frames = [Image.fromarray(frame) for frame in video_np]
                
                # Save as GIF
                gif_path = os.path.join(gif_dir, f"step_{t[0].item():04d}.gif")
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                              duration=100, loop=0)
                print(f"Saved GIF: {gif_path}")
        
        return z_t
    
    return injection_fn
def temporal_consistency_guidance(model, prev_frames, overlap_frames, w_r=1.0):
    """Reconstruction guidance from the paper (Equation 7)."""
    
    # Freeze VAE decoder
    for param in model.vae.decoder.parameters():
        param.requires_grad = False
    
    def guidance_fn(z_t, t):
        if prev_frames is None:
            return torch.zeros_like(z_t)
        
        z_t_grad = z_t.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Get x_0 prediction (denoised latent)
            noise_pred = model.denoise_fn(z_t_grad, t)
            x_recon = model.predict_start_from_noise(z_t_grad, t, noise_pred)
            
            if x_recon.dim() == 5:  # Video latent
                x_recon_overlap = x_recon[:, :, :overlap_frames]
            else:  # Image latent, adjust accordingly
                x_recon_overlap = x_recon
            x_pixel_overlap = model.vae.decode(x_recon_overlap / model.latent_scale)
            x_pixel_overlap = (x_pixel_overlap.clamp(-1, 1) + 1) / 2
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_pixel_overlap, prev_frames, reduction='mean')
            print(recon_loss.item())
            # Gradient w.r.t z_t
            grad_z_t = torch.autograd.grad(recon_loss, z_t_grad)[0]
        
        # Scale gradient: paper uses -w_r * alpha_t * grad
        # The factor of 1/2 is typically absorbed into w_r
        alpha_t = extract(model.alphas_cumprod, t, z_t.shape)
        
        # Negative gradient = minimize loss
        scaled_grad = -1*w_r * alpha_t * grad_z_t
        
        return scaled_grad.detach()
    
    return guidance_fn

# def temporal_consistency_guidance(model, prev_frames, overlap_frames, guidance_scale=1.0):
#     def guidance_fn(x_t, t):
#         if prev_frames is None:
#             return 0.0
#         x_t_grad = x_t.detach().requires_grad_(True)
#         noise_pred = model.denoise_fn(x_t_grad, t)
#         x_0_pred = model.predict_start_from_noise(x_t_grad, t, noise_pred)
#         current_first_latent = x_0_pred[:, :, :overlap_frames]
#         for param in model.vae.decoder.parameters():
#             param.requires_grad = False
#         current_first_pixels = model.vae.decode(current_first_latent / model.latent_scale)
#         current_first_pixels = (current_first_pixels.clamp(-1, 1) + 1) / 2
#         print(current_first_pixels.shape, prev_frames.shape)
#         loss = F.mse_loss(current_first_pixels, prev_frames)
#         print(loss.item())
#         grad = torch.autograd.grad(loss, x_t_grad)[0]
#         return -guidance_scale * grad.detach()
#     return guidance_fn

CHANNELS_TO_MODE = {1: 'L', 3: 'RGB', 4: 'RGBA'}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

def video_tensor_to_gif(tensor, path, duration = 500, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f == frames:
        return t
    if f > frames:
        return t[:, :frames]
    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, channels = 3, num_frames = 16, horizontal_flip = False, force_num_frames = True, exts = ['gif']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        seq_len = tensor.shape[1]
        if seq_len >= self.num_frames:
            start_idx = torch.randint(0, seq_len - self.num_frames + 1, (1,)).item()
            tensor = tensor[:, start_idx:start_idx + self.num_frames, :, :]
        else:
            pad_size = self.num_frames - seq_len
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_size))
        tensor = self.cast_num_frames_fn(tensor)
        return tensor

class Trainer(object):
    def __init__(self, diffusion_model, folder, *, ema_decay = 0.995, num_frames = 16, train_batch_size = 32, train_lr = 1e-4, train_num_steps = 100000, gradient_accumulate_every = 2, amp = False, step_start_ema = 2000, update_ema_every = 10, save_and_sample_every = 1000, results_folder = './results', num_sample_rows = 4, max_grad_norm = None):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames
        self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames)
        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training'
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(
            list(diffusion_model.denoise_fn.parameters()) + 
            list(diffusion_model.vae.parameters()), 
            lr=train_lr
        )
        self.step = 0
        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm
        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'opt': self.opt.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load'
            milestone = max(all_milestones)
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])
        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])

    def train(self, log_fn = noop):
        assert callable(log_fn)
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                frames = next(self.dl).cuda()
                with autocast(enabled = self.amp, device_type='cuda'):
                    diffusion_loss = self.model(frames)
                    frames_normalized = normalize_img(frames)
                    recon, mean, logvar = self.model.vae(frames_normalized)
                    recon_loss = F.mse_loss(recon, frames_normalized)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                    kl_loss = kl_loss / frames.numel()
                    vae_loss = recon_loss + 0.0001 * kl_loss
                    total_loss = diffusion_loss + 0.8 * vae_loss
                    self.scaler.scale(total_loss / self.gradient_accumulate_every).backward()
            log = {'loss': diffusion_loss.item(), 'vae_loss': vae_loss.item()}
            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.denoise_fn.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.vae.parameters(), self.max_grad_norm)
            self.scaler.step(self.opt)
            if self.step % 20 == 0:
                print(f'{self.step}: diff_loss={diffusion_loss.item():.4f}, vae_loss={vae_loss.item():.4f}')
            self.scaler.update()
            self.opt.zero_grad()
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step != 0 and (self.step + 1) % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)
                all_videos = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos = all_videos[0][0]
                one_gif = rearrange(all_videos, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / f'{milestone}.gif')
                video_tensor_to_gif(one_gif, video_path)
                self.save(milestone)
            log_fn(log)
            self.step += 1
        print('training completed')