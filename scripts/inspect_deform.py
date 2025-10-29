import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = '/home/nasr/SurgeSplam/data/C3VDv2/c1_transverse1_t4_v4/color'
OUT_DIR = '/tmp/deform_inspect'
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])
files = files[:200]  # sample first 200
print(f'Found {len(files)} frames (sampling up to 200)')

def load(f):
    im = Image.open(os.path.join(DATA_DIR, f)).convert('RGB')
    return np.array(im).astype(np.float32)/255.0

prev = None
stats = []
centroids = []
for idx, f in enumerate(files):
    im = load(f)
    gray = im.mean(axis=2)
    # centroid of bright areas
    mask = gray > (gray.mean() + 0.05)
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        centroid = np.array([xs.mean(), ys.mean()])
    else:
        centroid = np.array([np.nan, np.nan])
    centroids.append(centroid)

    if prev is None:
        prev = im
        continue
    diff = np.abs(im - prev)
    mean_diff = diff.mean()
    pct_diff = (diff.mean(axis=2) > 0.02).mean()
    # save a few diff visualizations
    if idx < 6:
        disp = (diff / diff.max()) if diff.max()>0 else diff
        plt.imsave(os.path.join(OUT_DIR, f'diff_{idx-1:04d}_{idx:04d}.png'), disp)
    stats.append((idx-1, idx, mean_diff, pct_diff))
    prev = im

stats = np.array(stats)
print('mean per-frame abs RGB diff (first 200):', stats[:,2].mean(), 'std:', stats[:,2].std())
print('mean per-frame pct pixels changed (>0.02):', stats[:,3].mean(), 'std:', stats[:,3].std())

# centroid shifts
centroids = np.array(centroids)
valid = ~np.isnan(centroids[:,0])
centroids_valid = centroids[valid]
deltas = np.linalg.norm(np.diff(centroids_valid, axis=0), axis=1)
print('centroid shift mean (pixels):', deltas.mean(), 'std:', deltas.std())

# save centroid trace visualization on first frame
try:
    base = load(files[0])
    plt.figure(figsize=(6,6))
    plt.imshow(base)
    pts = centroids_valid
    if pts.shape[0]>1:
        plt.plot(pts[:,0], pts[:,1], '-o', color='yellow', markersize=2)
    plt.title('Centroid trace')
    plt.axis('off')
    plt.savefig(os.path.join(OUT_DIR, 'centroid_trace.png'), bbox_inches='tight')
    plt.close()
    print('Saved visualizations to', OUT_DIR)
except Exception as e:
    print('Could not save centroid trace:', e)

# print a small per-frame table
print('\nSample per-frame stats (first 10):')
for r in stats[:10]:
    print(f'{r[0]}->{r[1]} mean_diff={r[2]:.5f} pct_changed={r[3]:.4f}')

# write a short summary file
with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as fh:
    fh.write(f'frames_sampled={len(files)}\n')
    fh.write(f'mean_abs_diff={stats[:,2].mean():.6f}\n')
    fh.write(f'mean_pct_changed={stats[:,3].mean():.6f}\n')
    fh.write(f'centroid_shift_mean={deltas.mean():.6f}\n')

print('Done')
