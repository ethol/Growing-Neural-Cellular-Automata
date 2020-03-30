# @title Imports and Notebook Utilities

import glob
import os
import common

import numpy as np
import tqdm

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp

# @title Training Progress (Checkpoints)
training_location = 'train_log_old'
models = []
for i in [100, 500, 1000, 4000, 8000]:
    ca = common.CAModel()
    ca.load_weights('train_log/%04d' % i)
    models.append(ca)

out_fn = 'movies/train_steps_damage_%d.mp4' % common.DAMAGE_N
x = np.zeros([len(models), 72, 72, common.CHANNEL_N], np.float32)
x[..., 36, 36, 3:] = 1.0
with common.VideoWriter(out_fn) as vid:
    for i in tqdm.trange(500):
        vis = np.hstack(common.to_rgb(x))
        vid.add(common.zoom(vis, 2))
        for ca, xk in zip(models, x):
            xk[:] = ca(xk[None, ...])[0]

frames = sorted(glob.glob('train_log/batches_*.jpg'))
mvp.ImageSequenceClip(frames, fps=10.0).write_videofile('movies/batches.mp4')

frames = sorted(glob.glob('train_log/*_pool.jpg'))[:800]
mvp.ImageSequenceClip(frames, fps=20.0).write_videofile('movies/pool.mp4')
