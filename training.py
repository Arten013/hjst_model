from hjst_model.config import LayerModelConfig
import os
from hjst_model.layers import Doc2VecLayer, WVAverageLayer, GDBWVAverageLayer

MODEL_DIR = './results/hjst/models/'
CONF_DIR = './configs'
TRAININGSET = 'tokai'
MODEL_CLASS = WVAverageLayer
LEVELS = ['Law', 'Article', 'Sentence']

layer_conf = LayerModelConfig()
layer_conf.link(os.path.join(CONF_DIR, 'layer.conf'), create_if_missing=True)
layer_conf.set_directory(MODEL_DIR, os.path.join(CONF_DIR, 'dataset.conf'))
layer_conf.update_file()
for l in LEVELS:
    layer_conf.create_layer(TRAININGSET, WVAverageLayer, l)
    layer_conf.create_layer(TRAININGSET, Doc2VecLayer, l)
