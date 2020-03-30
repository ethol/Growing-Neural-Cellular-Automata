import base64
import common
import glob
import zipfile
import json

import numpy as np

EMOJI = ['me', "trump", "corona"]


# EMOJI = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'
# EMOJI = '🦎'

def get_model(emoji='me', fire_rate=0.5, use_pool=1, damage_n=3, run=0,
              prefix='models_example/', output='model'):
    path = prefix
    assert fire_rate in [0.5, 1.0]
    if fire_rate == 0.5:
        path += 'use_sample_pool_%d damage_n_%d ' % (use_pool, damage_n)
    elif fire_rate == 1.0:
        path += 'fire_rate_1.0 '
    code = emoji.upper()
    path += 'target_%s run_index_%d/8000' % (code, run)
    assert output in ['model', 'json']
    print(path)
    if output == 'model':
        ca = common.CAModel(channel_n=16, fire_rate=fire_rate)
        ca.load_weights(path)

        return ca
    elif output == 'json':
        return open(path + '.json', 'r').read()


model = "CHECKPOINT"  # @param ['CHECKPOINT', '😀 1F600', '💥 1F4A5', '👁 1F441', '🦎 1F98E', '🐠 1F420', '🦋 1F98B', '🐞 1F41E', '🕸 1F578', '🥨 1F968', '🎄 1F384']
model_type = '3 regenerating'  # @param ['1 naive', '2 persistent', '3 regenerating']

# @markdown Shift-click to seed the pattern

if model != 'CHECKPOINT':
    code = model.split(' ')[1]
    emoji = chr(int(code, 16))
    experiment_i = int(model_type.split()[0]) - 1
    use_pool = (0, 1, 1)[experiment_i]
    damage_n = (0, 0, 3)[experiment_i]
    model_str = get_model(emoji, use_pool=use_pool, damage_n=damage_n, output='json')
else:
    last_checkpoint_fn = sorted(glob.glob('train_log/*.json'))[-1]
    model_str = open(last_checkpoint_fn).read()


# @title WebGL Demo

# @markdown This code exports quantized models for the WebGL demo that is used in the article.
# @markdown The demo code can be found at https://github.com/distillpub/post--growing-ca/blob/master/public/ca.js

def pack_layer(weight, bias, outputType=np.uint8):
    in_ch, out_ch = weight.shape
    assert (in_ch % 4 == 0) and (out_ch % 4 == 0) and (bias.shape == (out_ch,))
    weight_scale, bias_scale = 1.0, 1.0
    if outputType == np.uint8:
        weight_scale = 2.0 * np.abs(weight).max()
        bias_scale = 2.0 * np.abs(bias).max()
        weight = np.round((weight / weight_scale + 0.5) * 255)
        bias = np.round((bias / bias_scale + 0.5) * 255)
    packed = np.vstack([weight, bias[None, ...]])
    packed = packed.reshape(in_ch + 1, out_ch // 4, 4)
    packed = outputType(packed)
    packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
    return {'data_b64': packed_b64, 'in_ch': in_ch, 'out_ch': out_ch,
            'weight_scale': weight_scale, 'bias_scale': bias_scale,
            'type': outputType.__name__}


def export_ca_to_webgl_demo(ca, outputType=np.uint8):
    # reorder the first layer inputs to meet webgl demo perception layout
    chn = ca.channel_n
    w1 = ca.weights[0][0, 0].numpy()
    w1 = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3 * chn, -1)
    layers = [
        pack_layer(w1, ca.weights[1].numpy(), outputType),
        pack_layer(ca.weights[2][0, 0].numpy(), ca.weights[3].numpy(), outputType)
    ]
    return json.dumps(layers)


with zipfile.ZipFile('webgl_models8.zip', 'w') as zf:
    for e in EMOJI:
        zf.writestr('ex1_%s.json' % e, export_ca_to_webgl_demo(get_model(e, use_pool=0, damage_n=0)))
        zf.writestr('ex2_%s.json' % e, export_ca_to_webgl_demo(get_model(e, use_pool=1, damage_n=0)))
        zf.writestr('ex3_%s.json' % e, export_ca_to_webgl_demo(get_model(e, use_pool=1, damage_n=3)))
