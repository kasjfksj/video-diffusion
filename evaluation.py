import torch
import numpy as np
from scipy import linalg
from torchvision.models.video import r3d_18
import torch.nn as nn
import cv2

class FVDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # Use I3D or R3D for feature extraction
        self.model = r3d_18(pretrained=True).to(device)
        self.model.eval()
        # Remove the final classification layer
        self.model.fc = nn.Identity()
    
    def extract_features(self, videos):
        """
        videos: tensor of shape (N, T, C, H, W) or (N, C, T, H, W)
        Returns: features of shape (N, feature_dim)
        """
        features = []
        with torch.no_grad():
            for video in videos:
                # Ensure format is (C, T, H, W)
                if video.shape[0] != 3:
                    video = video.permute(1, 0, 2, 3)
                
                video = video.unsqueeze(0).to(self.device)
                feat = self.model(video)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet distance between two Gaussian distributions"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fvd
    
    def compute_fvd(self, videos_a, videos_b):
        """
        Compute FVD between two sets of videos
        videos_a, videos_b: lists or tensors of videos
        """
        features_a = self.extract_features(videos_a)
        features_b = self.extract_features(videos_b)
        
        mu_a = np.mean(features_a, axis=0)
        sigma_a = np.cov(features_a, rowvar=False)
        
        mu_b = np.mean(features_b, axis=0)
        sigma_b = np.cov(features_b, rowvar=False)
        
        fvd = self.calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
        return fvd
class FlowWarpingError:
    def __init__(self, flow_method='farneback'):
        self.flow_method = flow_method
    
    def compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow between two frames
        frame1, frame2: numpy arrays (H, W, C) or (H, W)
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        if self.flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        return flow
    
    def warp_frame(self, frame, flow):
        """
        Warp frame according to optical flow
        frame: torch tensor (C, H, W) or numpy array (H, W, C)
        flow: numpy array (H, W, 2)
        """
        h, w = flow.shape[:2]
        
        # Create coordinate grid
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w)
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
        
        # Add flow to get target coordinates
        flow_map = flow_map + flow
        
        # Warp using remap
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()
        
        warped = cv2.remap(
            frame, 
            flow_map[:, :, 0], 
            flow_map[:, :, 1],
            cv2.INTER_LINEAR
        )
        return warped
    
    def compute_fwe(self, video):
        """
        Compute Flow Warping Error for a video
        video: numpy array (T, H, W, C) or (T, C, H, W)
        Returns: average FWE across all frame pairs
        """
        if len(video.shape) == 4 and video.shape[1] == 3:
            # Convert from (T, C, H, W) to (T, H, W, C)
            video = np.transpose(video, (0, 2, 3, 1))
        
        video = (video * 255).astype(np.uint8) if video.max() <= 1.0 else video.astype(np.uint8)
        
        errors = []
        for t in range(len(video) - 1):
            frame1 = video[t]
            frame2 = video[t + 1]
            
            # Compute optical flow
            flow = self.compute_optical_flow(frame1, frame2)
            
            # Warp frame1 using flow
            warped = self.warp_frame(frame1, flow)
            
            # Compute error between warped frame and actual next frame
            error = np.mean((warped.astype(float) - frame2.astype(float)) ** 2)
            errors.append(error)
        
        return np.mean(errors)
    
    def compare_models(self, videos_a, videos_b):
        """Compare FWE between two sets of videos"""
        fwe_a = [self.compute_fwe(video) for video in videos_a]
        fwe_b = [self.compute_fwe(video) for video in videos_b]
        
        return {
            'model_a_mean': np.mean(fwe_a),
            'model_a_std': np.std(fwe_a),
            'model_b_mean': np.mean(fwe_b),
            'model_b_std': np.std(fwe_b)
        }

class TemporalFlickeringScore:
    def __init__(self):
        pass
    
    def compute_pixel_flickering(self, video):
        """
        Compute pixel-level flickering
        video: numpy array (T, H, W, C) or (T, C, H, W)
        """
        if len(video.shape) == 4 and video.shape[1] in [1, 3]:
            # Convert from (T, C, H, W) to (T, H, W, C)
            video = np.transpose(video, (0, 2, 3, 1))
        
        # Normalize to [0, 1]
        if video.max() > 1.0:
            video = video / 255.0
        
        # Compute frame differences
        diffs = np.diff(video, axis=0)
        
        # Average absolute difference per frame transition
        flickering = np.mean(np.abs(diffs))
        return flickering
    
    def compute_perceptual_flickering(self, video):
        """
        Compute perceptual flickering using gradients
        """
        if len(video.shape) == 4 and video.shape[1] in [1, 3]:
            video = np.transpose(video, (0, 2, 3, 1))
        
        if video.max() > 1.0:
            video = video / 255.0
        
        # Convert to grayscale for simpler analysis
        if video.shape[-1] == 3:
            gray = np.mean(video, axis=-1)
        else:
            gray = video.squeeze(-1)
        
        # Compute temporal gradient
        temporal_grad = np.diff(gray, axis=0)
        
        # Compute spatial gradients for each frame
        spatial_grads = []
        for frame in gray:
            gx = np.abs(np.diff(frame, axis=1))
            gy = np.abs(np.diff(frame, axis=0))
            spatial_grads.append(np.mean(gx) + np.mean(gy))
        
        # Flickering normalized by spatial complexity
        avg_spatial = np.mean(spatial_grads)
        flickering = np.mean(np.abs(temporal_grad)) / (avg_spatial + 1e-8)
        
        return flickering
    
    def compute_frequency_flickering(self, video):
        """
        Compute flickering in frequency domain
        """
        if len(video.shape) == 4 and video.shape[1] in [1, 3]:
            video = np.transpose(video, (0, 2, 3, 1))
        
        if video.max() > 1.0:
            video = video / 255.0
        
        # Convert to grayscale
        if video.shape[-1] == 3:
            gray = np.mean(video, axis=-1)
        else:
            gray = video.squeeze(-1)
        
        # Compute FFT along temporal dimension for each pixel
        T, H, W = gray.shape
        
        # Sample pixels to reduce computation
        sample_h = np.linspace(0, H-1, min(32, H), dtype=int)
        sample_w = np.linspace(0, W-1, min(32, W), dtype=int)
        
        high_freq_energy = []
        for h in sample_h:
            for w in sample_w:
                temporal_signal = gray[:, h, w]
                fft = np.fft.fft(temporal_signal)
                power = np.abs(fft) ** 2
                
                # High frequency energy (above Nyquist/2)
                high_freq = np.sum(power[T//4:T//2])
                total = np.sum(power[1:T//2])  # Exclude DC
                
                if total > 0:
                    high_freq_energy.append(high_freq / total)
        
        return np.mean(high_freq_energy)
    
    def compute_tfs(self, video, method='all'):
        """
        Compute Temporal Flickering Score
        method: 'pixel', 'perceptual', 'frequency', or 'all'
        """
        scores = {}
        
        if method in ['pixel', 'all']:
            scores['pixel'] = self.compute_pixel_flickering(video)
        
        if method in ['perceptual', 'all']:
            scores['perceptual'] = self.compute_perceptual_flickering(video)
        
        if method in ['frequency', 'all']:
            scores['frequency'] = self.compute_frequency_flickering(video)
        
        if method == 'all':
            scores['combined'] = np.mean(list(scores.values()))
        
        return scores
    
    def compare_models(self, videos_a, videos_b, method='all'):
        """Compare TFS between two sets of videos"""
        tfs_a = [self.compute_tfs(video, method) for video in videos_a]
        tfs_b = [self.compute_tfs(video, method) for video in videos_b]
        
        results = {}
        if method == 'all':
            for key in tfs_a[0].keys():
                results[key] = {
                    'model_a_mean': np.mean([s[key] for s in tfs_a]),
                    'model_a_std': np.std([s[key] for s in tfs_a]),
                    'model_b_mean': np.mean([s[key] for s in tfs_b]),
                    'model_b_std': np.std([s[key] for s in tfs_b])
                }
        else:
            results[method] = {
                'model_a_mean': np.mean([s[method] for s in tfs_a]),
                'model_a_std': np.std([s[method] for s in tfs_a]),
                'model_b_mean': np.mean([s[method] for s in tfs_b]),
                'model_b_std': np.std([s[method] for s in tfs_b])
            }
        
        return results
    

