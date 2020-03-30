# @title Imports and Notebook Utilities

import json
import os

import matplotlib.pylab as pl
import numpy as np
import tensorflow as tf
from IPython.display import clear_output

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import common

# @title Train Utilities (SamplePool, Model Export, Damage)
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

url = 'C:/Users/tomglove_b/Documents/GitHub/playground/input/corona2.png'

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = tf.cast(x * x + y * y < 1.0, tf.float32)
    return mask


def export_model(ca, base_fn):
    ca.save_weights(base_fn)

    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, common.CHANNEL_N]),
        fire_rate=tf.constant(0.5),
        angle=tf.constant(0.0),
        step_size=tf.constant(1.0))
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
        'format': 'graph-model',
        'modelTopology': graph_json,
        'weightsManifest': [],
    }
    with open(base_fn + '.json', 'w') as f:
        json.dump(model_json, f)


def generate_pool_figures(pool, step_i):
    tiled_pool = common.tile2d(common.to_rgb(pool.x[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
    common.imwrite('train_log/%04d_pool.jpg' % step_i, tiled_pool)


def visualize_batch(x0, x, step_i):
    vis0 = np.hstack(common.to_rgb(x0).numpy())
    vis1 = np.hstack(common.to_rgb(x).numpy())
    vis = np.vstack([vis0, vis1])
    common.imwrite('train_log/batches_%04d.jpg' % step_i, vis)
    print('batch (before/after):')
    common.imshow(vis)


def plot_loss(loss_log):
    pl.figure(figsize=(10, 4))
    pl.title('Loss history (log10)')
    pl.plot(np.log10(loss_log), '.', alpha=0.1)
    pl.show()


target_img = common.load_emoji(url)
common.imshow(common.zoom(common.to_rgb(target_img), 2), fmt='png')

p = common.TARGET_PADDING
pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
seed = np.zeros([h, w, common.CHANNEL_N], np.float32)
seed[h // 2, w // 2, 3:] = 1.0


def loss_f(x):
    return tf.reduce_mean(tf.square(common.to_rgba(x) - pad_target), [-2, -3, -1])


ca = common.CAModel()

loss_log = []

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr * 0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

loss0 = loss_f(seed).numpy()
pool = SamplePool(x=np.repeat(seed[None, ...], common.POOL_SIZE, 0))


# !mkdir -p train_log && rm -f train_log/*

@tf.function
def train_step(x):
    iter_n = tf.random.uniform([], 64, 96, tf.int32)
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            x = ca(x)
        loss = tf.reduce_mean(loss_f(x))
    grads = g.gradient(loss, ca.weights)
    grads = [g / (tf.norm(g) + 1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss


for i in range(8000 + 1):
    if common.USE_PATTERN_POOL:
        batch = pool.sample(common.BATCH_SIZE)
        x0 = batch.x
        loss_rank = loss_f(x0).numpy().argsort()[::-1]
        x0 = x0[loss_rank]
        x0[:1] = seed
        if common.DAMAGE_N:
            damage = 1.0 - make_circle_masks(common.DAMAGE_N, h, w).numpy()[..., None]
            x0[-common.DAMAGE_N:] *= damage
    else:
        x0 = np.repeat(seed[None, ...], common.BATCH_SIZE, 0)

    x, loss = train_step(x0)

    if common.USE_PATTERN_POOL:
        batch.x[:] = x
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    if step_i % 10 == 0:
        generate_pool_figures(pool, step_i)
    if step_i % 100 == 0:
        clear_output()
        visualize_batch(x0, x, step_i)
        # plot_loss(loss_log)
        export_model(ca, 'train_log/%04d' % step_i)

    print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(loss)), end='')
