import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import from your video_flow.py file
from video_flow import VelocityNet, FlowMatching

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_gif(frames, filename, duration=100):
    """Save frames as GIF"""
    pil_frames = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    
    pil_frames[0].save(filename, save_all=True, append_images=pil_frames[1:], 
                       duration=duration, loop=0)
    print(f"Saved: {filename}")


def save_frame_as_image(frame, filename):
    """Save a single frame as PNG image"""
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    Image.fromarray(frame).save(filename)
    print(f"Saved: {filename}")


# Load trained model
print("Loading model...")
velocity_net = VelocityNet().to(DEVICE)
flow_model = FlowMatching(velocity_net).to(DEVICE)

checkpoint = torch.load('flow_matching_video_final.pth', map_location=DEVICE, weights_only=True)
velocity_net.load_state_dict(checkpoint['model'])
velocity_net.eval()

# Load camera motion values from text file
print("Loading camera motion...")
camera_motion_values = np.loadtxt('camera_motion.txt')
print(f"Loaded {len(camera_motion_values)} camera motion values")
print(f"Camera motion range: [{camera_motion_values.min():.4f}, {camera_motion_values.max():.4f}]")

# Load initial frame from data
data = np.load('0.npz')
video = data['video']
initial_frame = torch.from_numpy((video[0].astype(np.float32) / 255.0) * 2 - 1)
initial_frame = initial_frame.unsqueeze(0).to(DEVICE)

# Generate video sequence
num_frames = min(16, len(camera_motion_values))  # Don't exceed available motion data

num_frames=200
print(f"Generating {num_frames} frames...")
generated_frames = [video[0]]  # Start with original first frame

# Save first frame
save_frame_as_image(generated_frames[0], 'frame_0.png')

current_frame = initial_frame  # Already in [-1, 1] range

for i in tqdm(range(num_frames - 1)):
    # Use the camera motion value for this frame
    # Frame i+1 uses motion value at index i (motion from frame i to i+1)
    motion_value = camera_motion_values[i]
    print(motion_value)
    cond = torch.tensor([[motion_value]], dtype=torch.float32, device=DEVICE)
    
    if i == 0:
        print(f"First motion value: {motion_value:.4f}, cond shape: {cond.shape}")
    
    with torch.no_grad():
        # Pass current frame as prev_frame
        next_frame = flow_model.sample(
            prev_frame=current_frame,
            cond=cond,
            num_steps=50,
            method='euler'
        )
    
    generated_frames.append(next_frame[0].cpu().numpy())
    
    # Save second frame
    if i == 0:
        save_frame_as_image(generated_frames[1], 'frame_1.png')
    
    # Update current frame (convert back to [-1, 1])
    current_frame = next_frame.float() / 255.0
    current_frame = (current_frame * 2) - 1

# Save as GIF
save_gif(generated_frames, 'generated_video_2.gif', duration=200)

print("\nDone! Check:")
print("  - frame_0.png (first frame)")
print("  - frame_1.png (second frame)")
print("  - generated_video.gif (full animation)")