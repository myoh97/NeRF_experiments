import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

import json
import os
import numpy as np

basedir = '/root/dataset/NeRF/nerf_synthetic/bed'
with open(os.path.join(basedir, 'transforms_train.json'), 'r') as fp:
            meta = json.load(fp)
poses = []

for frame in meta['frames']:
    p = np.array(frame['transform_matrix'])
    p = np.linalg.inv(p)
    poses.append(p)
poses = np.array(poses)
dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
origins = poses[:, :3, -1]

ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
_ = ax.quiver(
  origins[..., 0].flatten(),
  origins[..., 1].flatten(),
  origins[..., 2].flatten(),
  dirs[..., 0].flatten(),
  dirs[..., 1].flatten(),
  dirs[..., 2].flatten(), length=0.5, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig('dir.png')