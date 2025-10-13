from PIL import Image
import numpy as np

# Replace with your actual file path
path = "experiments/C3VD_base/trans_t1_a/eval/color/color_0007.png"

img = Image.open(path).convert("RGB")  # ensure it's RGB
arr = np.array(img) / 255.0  # normalize to [0,1]
print("Mean pixel value:", arr.mean())
print("Min:", arr.min(), "Max:", arr.max())
