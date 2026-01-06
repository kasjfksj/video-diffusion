import numpy as np
from PIL import Image

def gif_to_npz(gif_path, npz_path='output.npz'):
    """
    Convert a GIF to an NPZ file containing a NumPy array of frames.
    
    Args:
        gif_path (str): Path to the input GIF file.
        npz_path (str): Path to save the output NPZ file (default: 'output.npz').
    """
    # Load GIF and extract frames
    with Image.open(gif_path) as img:
        frames = []
        try:
            # Iterate through all frames (handles animated GIFs)
            while True:
                # Convert to RGB (GIFs may have palette/transparency)
                frame = img.convert('RGB')
                # Convert to NumPy array (H, W, C)
                frame_array = np.array(frame)
                frames.append(frame_array)
                img.seek(img.tell() + 1)  # Move to next frame
        except EOFError:
            # End of frames
            pass
    
    # Stack into 4D array: (num_frames, height, width, 3)
    frames_array = np.stack(frames, axis=0)
    
    # Save to NPZ (compressed by default)
    np.savez(npz_path, frames=frames_array)
    print(f"Saved {len(frames)} frames to {npz_path}")
    print(f"Array shape: {frames_array.shape}")

# Example usage
gif_to_npz('/home/lu/Documents/video-diffusion-pytorch/video_diffusion_pytorch/data/0.gif')  # Replace with your GIF path