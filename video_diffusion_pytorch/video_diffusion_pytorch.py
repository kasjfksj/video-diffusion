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
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM

# helpers functions

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
    # Concatenate: [1,3,4,64,64] + [1,3,10,64,64] → [1,3,14,64,64]

    frames = torch.cat([prev_frames, current_frames], dim=2)
    
    # To (14, 64, 64, 3) and uint8
    frames = frames.squeeze(0).permute(1, 2, 3, 0)  # [14,64,64,3]
    frames = (frames.clamp(0, 1) * 255).byte().cpu()
    
    # Save as 0.gif
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

# ========== VAE Components ==========

class VAEEncoder(nn.Module):
    """Encoder that compresses video frames to latent space"""
    def __init__(self, channels=3, latent_dim=4, base_dim=32):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        
        # Spatial downsampling layers (8x8x downsampling total)
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
        # FIXED: Clamp logvar for stability
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar


class VAEDecoder(nn.Module):
    """Decoder that reconstructs video frames from latent space"""
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
            # REMOVED Tanh - output raw values, will normalize outside
        )
        
    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """Variational Autoencoder for video compression with proper scaling"""
    def __init__(self, channels=3, latent_dim=4, base_dim=64):
        super().__init__()
        self.encoder = VAEEncoder(channels, latent_dim, base_dim)
        self.decoder = VAEDecoder(latent_dim, channels, base_dim)
        self.latent_dim = latent_dim
        
        # CRITICAL: Latent scaling factors (learned during VAE training)
        # These ensure latents have unit variance for diffusion
        self.register_buffer('latent_scale', torch.ones(1))
        self.register_buffer('latent_shift', torch.zeros(1))
        
    def encode(self, x):
        """Encode to latent distribution"""
        mean, logvar = self.encoder(x)
        return mean, logvar
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode_to_latent(self, x):
        """Encode and return scaled latent (for diffusion training)"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        # Scale to unit variance
        z_scaled = (z - self.latent_shift) / self.latent_scale
        return z_scaled
    
    def decode_from_latent(self, z_scaled):
        """Decode from scaled latent (for diffusion sampling)"""
        # Unscale first
        z = z_scaled * self.latent_scale + self.latent_shift
        return self.decode(z)
    
    def forward(self, x):
        """Full VAE forward pass"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
    
    @torch.no_grad()
    def compute_latent_stats(self, dataloader, num_batches=100):
        """Compute mean and std of latent space for scaling"""
        print("Computing latent statistics for scaling...")
        self.eval()
        
        latents = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                x = batch['current'].cuda()
            else:
                x = batch.cuda()
            
            # Normalize input
            x = normalize_img(x)
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            latents.append(z)
        
        latents = torch.cat(latents, dim=0)
        
        # Compute statistics
        latent_mean = latents.mean()
        latent_std = latents.std()
        
        print(f"Latent mean: {latent_mean.item():.4f}, std: {latent_std.item():.4f}")
        
        self.latent_shift.copy_(latent_mean)
        self.latent_scale.copy_(latent_std)
        
        self.train()
# ========== Previous Frame Conditioning ==========

class FrameConditioningEncoder(nn.Module):
    """Encodes previous frames' latents for cross-attention conditioning"""
    def __init__(self, latent_dim=4, feature_dim=512, num_prev_frames=4):
        super().__init__()
        self.num_prev_frames = num_prev_frames
        
        # Extract rich spatial-temporal features
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(latent_dim, 128, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(128, 256, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(256, feature_dim, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.SiLU(),
        )
        
    def forward(self, prev_latents):
        """
        Args:
            prev_latents: (B, C, F, H, W) - latents of previous frames
        Returns:
            features: (B, F*H*W, feature_dim) - spatial-temporal features for cross-attention
        """
        # Extract features: [B, C, F, H, W] -> [B, feature_dim, F, H', W']

        features = self.feature_extractor(prev_latents)
        
        # Reshape for cross-attention: [B, feature_dim, F, H', W'] -> [B, F*H'*W', feature_dim]
        b, c, f, h, w = features.shape
        features = rearrange(features, 'b c f h w -> b (f h w) c')
        return features


# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
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
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

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
class MultiScaleFrameEncoder(nn.Module):
    def __init__(self, latent_dim=4, num_prev_frames=4):
        super().__init__()
        
        # Extract features at multiple scales
        self.scale_encoders = nn.ModuleList([
            self._make_scale_encoder(latent_dim, 128, stride=1),   # Fine
            self._make_scale_encoder(latent_dim, 256, stride=2),   # Medium
            self._make_scale_encoder(latent_dim, 512, stride=4),   # Coarse
        ])
        
    def _make_scale_encoder(self, in_dim, out_dim, stride):
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim // 2, (1, 3, 3), (1, stride, stride), (0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(out_dim // 2, out_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
    
    def forward(self, prev_latents):
        """Returns list of features at different scales"""
        return [encoder(prev_latents) for encoder in self.scale_encoders]
def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x, context=None):
        """
        Args:
            x: (B, ..., query_dim) - queries from current features
            context: (B, seq_len, context_dim) - keys/values from previous frames
        """
        # Handle 5D input: (B, C, F, H, W) -> (B, F*H*W, C)
        if x.ndim == 5:
            b, c, f, h, w = x.shape
            x = rearrange(x, 'b c f h w -> b (f h w) c')
            reshape_back = True
        else:
            reshape_back = False
            b = x.shape[0]
        
        context = default(context, x)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        # Attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if reshape_back:
            out = rearrange(out, 'b (f h w) c -> b c f h w', f=f, h=h, w=w)
        
        return out
# attention along space and time

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
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        # FIXED: Remove early return that bypasses attention
        # if exists(focus_present_mask) and focus_present_mask.all():
        #     values = qkv[-1]
        #     return self.to_out(values)

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        q = q * self.scale

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # FIXED: Better numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1, dtype=torch.float32)
        
        # Convert back if needed
        if attn.dtype != v.dtype:
            attn = attn.to(v.dtype)

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 4,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        block_type = 'resnet',
        frame_cond_dim = 512
    ):
        super().__init__()
        self.channels = channels

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32)

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 8
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        
        # IMPROVED: Frame conditioning with learned projection
        self.has_frame_cross_attn = exists(frame_cond_dim)
        if self.has_frame_cross_attn:
            # Create cross-attention modules for each resolution
            self.frame_cross_attns_down = nn.ModuleList([])
            self.frame_cross_attns_up = nn.ModuleList([])
            
            for dim_out in dims[1:]:  # Skip init_dim
                self.frame_cross_attns_down.append(
                    Residual(PreNorm(dim_out, 
                        CrossAttention(dim_out, context_dim=frame_cond_dim, 
                                     heads=attn_heads, dim_head=attn_dim_head)))
                )
            
            for dim_in in reversed(dims[:-1]):
                self.frame_cross_attns_up.append(
                    Residual(PreNorm(dim_in,
                        CrossAttention(dim_in, context_dim=frame_cond_dim,
                                     heads=attn_heads, dim_head=attn_dim_head)))
                )

        # FIXED: Don't add frame_cond_dim to cond_dim, it's processed separately
        cond_dim = time_dim + int(cond_dim or 0)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        block_klass = ResnetBlock
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        frame_cond_scale = 1.,
        **kwargs
    ):
        """Classifier-free guidance for both text and frame conditioning"""
        logits = self.forward(*args, null_cond_prob = 0., null_frame_cond_prob = 0., **kwargs)
        
        if cond_scale == 1 and frame_cond_scale == 1:
            return logits
        

        # Apply text conditioning CFG FIRST (if enabled)
        if self.has_cond and cond_scale != 1:
            null_logits = self.forward(*args, null_cond_prob = 1., null_frame_cond_prob = 0., **kwargs)
            logits = null_logits + (logits - null_logits) * cond_scale
        # Then apply frame conditioning CFG
        if self.has_frame_cross_attn and frame_cond_scale != 1:
            null_frame_logits = self.forward(*args, null_cond_prob = 0., null_frame_cond_prob = 1., **kwargs)
            logits = null_frame_logits + (logits - null_frame_logits) * frame_cond_scale

        return logits

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
         frame_cond_features=None,
        null_frame_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)
        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if self.has_cond:
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1)
        if self.has_frame_cross_attn and exists(frame_cond_features):
            if null_frame_cond_prob > 0:
                # Create mask for nullifying frame conditioning
                mask = prob_mask_like((batch,), null_frame_cond_prob, device = device)
                # Expand mask to match feature dimensions: [B] -> [B, seq_len, feature_dim]
                mask_expanded = mask[:, None, None].float()
                # Zero out features where mask is True
                frame_cond_features = frame_cond_features * (1 - mask_expanded)



        h = []
        frame_attn_idx = 0
        
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            
            # Add cross-attention to previous frames
            if self.has_frame_cross_attn and exists(frame_cond_features):
                x = self.frame_cross_attns_down[frame_attn_idx](x, context=frame_cond_features)
                frame_attn_idx += 1
            
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        frame_attn_idx = 0
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            
            # Add cross-attention to previous frames
            if self.has_frame_cross_attn and exists(frame_cond_features):
                x = self.frame_cross_attns_up[frame_attn_idx](x, context=frame_cond_features)
                frame_attn_idx += 1
            
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def frame_consistency_guidance(x, prev_frames, cond, metric='l2'):
    """
    Guidance function that minimizes distance between the first frame of x 
    and the last frame of prev_frames (both in pixel space).
    
    Args:
        x: (B, C, F, H, W) - generated frames in pixel space [0, 1]
        prev_frames: (B, C, num_prev_frames, H, W) - previous frames in pixel space [0, 1]
        cond: Text conditioning (unused)
        metric: 'l2' for MSE or 'l1' for MAE
    
    Returns:
        Scalar loss value
    """
    if prev_frames is None:
        return torch.tensor(0.0, device=x.device)
    x_first = x[:, :, 0, :, :]  
    prev_last = prev_frames[:, :, -1, :, :] 
    if metric == 'l2':
        loss = F.mse_loss(x_first, prev_last)
    elif metric == 'l1':
        loss = F.l1_loss(x_first, prev_last)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return loss
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        vae,
        image_size,
        num_frames,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
        latent_scale=1.0,
        frame_cond_encoder=None  # ADD THIS
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.vae = vae
        self.frame_cond_encoder = frame_cond_encoder  # ADD THIS
        self.latent_scale = latent_scale
        
        # FIX: Freeze VAE during diffusion training
        for param in self.vae.parameters():
            param.requires_grad = False
        
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

    @torch.inference_mode()
    def p_sample(self, x, t, frame_cond_features=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        
        # Predict noise
        predicted_noise = self.denoise_fn(
            x, 
            t, 
            frame_cond_features=frame_cond_features,
            null_frame_cond_prob=0.  # No dropout during sampling
        )
        
        x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)

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

        model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape):
        """Generate samples unconditionally in latent space"""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    @torch.inference_mode()
    def sample(self, batch_size=16, prev_frames=None, frame_cond_scale=1.0):
        """
        Generate video frames with optional frame conditioning
        Args:
            batch_size: Number of samples
            prev_frames: Previous frames (B, C, F_prev, H, W) in [0, 1]
            frame_cond_scale: Guidance scale for frame conditioning
        Returns:
            samples: Generated frames in pixel space [0, 1]
        """
        device = next(self.denoise_fn.parameters()).device
        
        # Get latent space dimensions
        latent_size = self.image_size // 8
        latent_channels = self.vae.latent_dim
        num_frames = self.num_frames
        
        # Encode previous frames if provided
        frame_cond_features = None
        if exists(prev_frames) and exists(self.frame_cond_encoder):
            self.vae.eval()
            with torch.no_grad():
                prev_normalized = prev_frames * 2.0 - 1.0
                prev_normalized = prev_normalized.clamp(-1, 1)
                prev_latents, _ = self.vae.encode(prev_normalized)
                prev_latents = prev_latents * self.latent_scale
            
            frame_cond_features = self.frame_cond_encoder(prev_latents)
        
        # Start with random noise
        img = torch.randn((batch_size, latent_channels, num_frames, latent_size, latent_size), 
                        device=device)
        
        # Denoise with classifier-free guidance if frame conditioning is used
        for i in tqdm(reversed(range(0, self.num_timesteps)), 
                    desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            if exists(frame_cond_features) and frame_cond_scale != 1.0:
                # Conditional prediction
                noise_cond = self.denoise_fn(
                    img, t, 
                    frame_cond_features=frame_cond_features,
                    null_frame_cond_prob=0.
                )
                
                # Unconditional prediction
                noise_uncond = self.denoise_fn(
                    img, t,
                    frame_cond_features=frame_cond_features,
                    null_frame_cond_prob=1.0  # Drop conditioning
                )
                
                # Classifier-free guidance
                noise_pred = noise_uncond + frame_cond_scale * (noise_cond - noise_uncond)
                
                # Manual denoising step
                x_recon = self.predict_start_from_noise(img, t, noise_pred)
                x_recon = x_recon.clamp(-1, 1)
                model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=img, t=t)
                noise = torch.randn_like(img) if i > 0 else torch.zeros_like(img)
                img = model_mean + (0.5 * model_log_variance).exp() * noise
            else:
                img = self.p_sample(img, t, frame_cond_features=frame_cond_features)
        
        # Decode latents to pixels
        self.vae.eval()
        with torch.no_grad():
            samples = self.vae.decode(img / self.latent_scale)
            samples = samples.clamp(-1, 1)
            samples = (samples + 1) / 2
        
        return samples

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, prev_frames=None, noise=None, null_frame_cond_prob=0., 
                prob_focus_present=0., focus_present_mask=None):  # ADD THESE
        """
        Training loss with frame conditioning
        Args:
            x_start: Current frames in pixel space (B, C, F, H, W) in [0, 1]
            t: Timesteps (B,)
            prev_frames: Previous frames (B, C, F_prev, H, W) in [0, 1]
            noise: Optional noise (if None, sampled randomly)
            null_frame_cond_prob: Probability of dropping frame conditioning
            prob_focus_present: Probability of focusing on present frame (unused for now)
            focus_present_mask: Mask for focusing on present (unused for now)
        """
        b, c, f, h, w, device = *x_start.shape, x_start.device
        
        # Normalize input to [-1, 1]
        x_start_normalized = x_start * 2.0 - 1.0
        x_start_normalized = x_start_normalized.clamp(-1, 1)
        
        # Encode with VAE
        self.vae.eval()
        with torch.no_grad():
            x_start_latent, _ = self.vae.encode(x_start_normalized)
        
        # Scale latents
        x_start_latent = x_start_latent * self.latent_scale
        x_start_latent = x_start_latent.clamp(-10, 10)
        
        # Encode previous frames if provided
        frame_cond_features = None
        if exists(prev_frames) and exists(self.frame_cond_encoder):
            with torch.no_grad():
                prev_normalized = prev_frames * 2.0 - 1.0
                prev_normalized = prev_normalized.clamp(-1, 1)
                prev_latents, _ = self.vae.encode(prev_normalized)
                prev_latents = prev_latents * self.latent_scale
            
            # Extract conditioning features
            frame_cond_features = self.frame_cond_encoder(prev_latents)
        
        # Sample random noise
        noise = default(noise, lambda: torch.randn_like(x_start_latent))
        
        # Add noise to latent
        x_noisy = self.q_sample(x_start=x_start_latent, t=t, noise=noise)
        
        # Predict noise with frame conditioning
        predicted_noise = self.denoise_fn(
            x_noisy, 
            t, 
            frame_cond_features=frame_cond_features,
            null_frame_cond_prob=null_frame_cond_prob,
            prob_focus_present=prob_focus_present,  # Pass through
            focus_present_mask=focus_present_mask   # Pass through
        )
        
        # Loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    def forward(self, x, prev_frames=None, null_frame_cond_prob=0., *args, **kwargs):
        """
        Forward pass for training
        Args:
            x: Video frames (B, C, F, H, W) in pixel space [0, 1]
            prev_frames: Previous frames (B, C, F_prev, H, W) in [0, 1]
            null_frame_cond_prob: Probability of dropping frame conditioning
        """
        b, device, img_size = x.shape[0], x.device, self.image_size
        
        # Clamp input to [0, 1] range
        x = x.clamp(0, 1)
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        return self.p_losses(x, t, prev_frames=prev_frames, 
                            null_frame_cond_prob=null_frame_cond_prob, 
                            *args, **kwargs)


# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

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

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
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
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        num_prev_frames = 4,  # NEW
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.num_prev_frames = num_prev_frames  # NEW
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames + num_prev_frames) if force_num_frames else identity

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
        total_needed = self.num_prev_frames + self.num_frames
        seq_len = tensor.shape[1]  # current number of frames (T)

        if seq_len >= total_needed:
            # Randomly sample a contiguous block
            start_idx = torch.randint(0, seq_len - total_needed + 1, (1,)).item()
            tensor = tensor[:, start_idx:start_idx + total_needed, :, :]
        else:
            # If too short, pad with zeros (or repeat, or reflect)
            pad_size = total_needed - seq_len
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_size))
        tensor = self.cast_num_frames_fn(tensor)
        # Split into previous frames and current frames
        if tensor.shape[1] >= self.num_frames + self.num_prev_frames:
            prev_frames = tensor[:, :self.num_prev_frames]
            current_frames = tensor[:, self.num_prev_frames:self.num_prev_frames + self.num_frames]
            return {'current': current_frames, 'prev': prev_frames}
        else:
            # If not enough frames, pad with zeros for previous frames
            current_frames = tensor[:, :self.num_frames]
            prev_frames = torch.zeros(self.channels, self.num_prev_frames, self.image_size, self.image_size)
            return {'current': current_frames, 'prev': prev_frames}

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        num_prev_frames = 4,  # NEW
        train_batch_size = 32,
        train_lr = 1e-4,
        vae_lr = 1e-4,  # NEW: separate LR for VAE
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 2,
        max_grad_norm = None,
        null_frame_cond_prob = 0.1,  # NEW: probability of dropping frame conditioning
        frame_cond_scale = 1.5  # NEW: frame conditioning guidance scale
    ):
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
        self.null_frame_cond_prob = null_frame_cond_prob  # NEW
        self.frame_cond_scale = frame_cond_scale  # NEW

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames, num_prev_frames = num_prev_frames)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        
        # Separate optimizers for diffusion model and VAE
        self.opt = Adam(
            list(diffusion_model.denoise_fn.parameters()) + 
            list(diffusion_model.vae.parameters()) + 
            list(diffusion_model.frame_cond_encoder.parameters()), 
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
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])
        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])

    def get_n_conditioning_clips(self,n: int):
        """
        Returns a tensor of shape [n, C, T_prev, H, W] on the current device.
        It pulls the required number of clips from the cyclic DataLoader.
        """
        clips = []
        remaining = n
        while remaining > 0:
            batch = next(self.dl)                 # dict with 'prev' key
            prev = batch['prev'].cuda()           # [B, C, T_prev, H, W]
            take = min(remaining, prev.shape[0])
            clips.append(prev[:take])
            remaining -= take
        return torch.cat(clips, dim=0)
    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                
                data_dict = next(self.dl)
                current_frames = data_dict['current'].cuda()
                prev_frames = data_dict['prev'].cuda()

                with autocast(enabled = self.amp):
                    # Diffusion loss
                    diffusion_loss = self.model(
                        current_frames,
                        prev_frames=prev_frames,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask,
                        null_frame_cond_prob = self.null_frame_cond_prob
                    )

                    # save_gif(prev_frames,current_frames)
                    # VAE reconstruction loss
                    current_normalized = normalize_img(current_frames)
                    recon, mean, logvar = self.model.vae(current_normalized)
                    recon_loss = F.mse_loss(recon, current_normalized)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                    kl_loss = kl_loss / current_frames.numel()
                    vae_loss = recon_loss + 0.0001 * kl_loss  # Weight KL term

                    # Total loss
                    total_loss = diffusion_loss + 0.1 * vae_loss  # Weight VAE loss
                    
                    self.scaler.scale(total_loss / self.gradient_accumulate_every).backward()


            log = {'loss': diffusion_loss.item(), 'vae_loss': vae_loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)

                nn.utils.clip_grad_norm_(self.model.denoise_fn.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.vae.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.frame_cond_encoder.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            if self.step%20==0:
                print(f'{self.step}: diff_loss={diffusion_loss.item():.4f}, vae_loss={vae_loss.item():.4f}')
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step != 0 and (self.step+1 ) % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                

                # Sample with and without frame conditioning
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                # Get some previous frames from dataset for conditioning
                sample_prev = self.get_n_conditioning_clips(num_samples)  # Get enough samples
                # print(sample_prev.shape)
                # Sample with frame conditioning - just one batch, no need for multiple
                all_videos_cond = self.ema_model.sample(
                    batch_size=self.num_sample_rows ** 2, 
                    prev_frames=sample_prev,
                    frame_cond_scale=self.frame_cond_scale
                )
                all_videos_with_context = torch.cat([sample_prev, all_videos_cond], dim=2)
                def make_grid_gif(tensor, path, pad=2):
                    """
                    tensor : [N, C, T, H, W]  (float in [0,1])
                    """
                    # optional visual border
                    if pad > 0:
                        tensor = F.pad(tensor, (pad, pad, pad, pad))

                    # rearrange into a square grid
                    tensor = rearrange(
                        tensor,
                        '(i j) c f h w -> c f (i h) (j w)',
                        i=self.num_sample_rows,
                    )                                                   # [C, T, H_grid, W_grid]

                    video_tensor_to_gif(tensor.clamp(0, 1), path)

                # ---- 4a. GIF with both conditioning + generated frames ----
                gif_path_ctx = self.results_folder / f'{milestone}_cond_with_context.gif'
                make_grid_gif(all_videos_with_context, gif_path_ctx)



                # Sample without frame conditioning (unconditional)
                all_videos_uncond = list(map(
                    lambda n: self.ema_model.sample(batch_size=n), 
                    batches
                ))
                all_videos_uncond = torch.cat(all_videos_uncond, dim = 0)
                all_videos_uncond = F.pad(all_videos_uncond, (2, 2, 2, 2))
                one_gif_uncond = rearrange(all_videos_uncond, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path_uncond = str(self.results_folder / f'{milestone}_unconditioned.gif')
                video_tensor_to_gif(one_gif_uncond, video_path_uncond)
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')