import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def _to_display_rgb(frames: np.ndarray) -> np.ndarray:
    """Convert stored camera tensors to uint8 RGB for matplotlib."""
    if frames.dtype == np.uint8:
        return frames
    frames = frames.astype(np.float32)
    if frames.max() <= 1.0 and frames.min() >= 0.0:
        return (frames * 255.0).astype(np.uint8)
    # Legacy mean-subtracted observations: stretch to full range for display only.
    lo, hi = frames.min(), frames.max()
    return ((frames - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


with h5py.File("datasets/ik_expert_demos_vis.hdf5", "r") as f:
    frames = _to_display_rgb(f["data/demo_0/camera/table_cam"][:])

fig, ax = plt.subplots()
im = ax.imshow(frames[0])
ax.axis("off")

def update(i):
    im.set_array(frames[i])
    ax.set_title(f"step {i}")
    return [im]

ani = FuncAnimation(fig, update, frames=len(frames), interval=33)  # ~30 fps
plt.show()