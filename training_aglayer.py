from hjst_model.config import LayerModelConfig
import os
import hjst_model.layers
import re
import argparse
from jstatutree.etypes import *

parser = argparse.ArgumentParser(description='Training layer using kvs dataset.')
parser.add_argument('model',
                    help='training layer model')
parser.add_argument('base_layer',
                    help='training layer model')
parser.add_argument('--levels', nargs="+",
                    help='layer level')
parser.add_argument('--name', '-n', default='None',
                    help='model name')
parser.add_argument('--test', default=False, action="store_true",
                    help='debug mode')
#parser.add_argument('--params', '-p', nargs='*',
#        help='optional model parameter (format key:value)')
args = parser.parse_args()

test_dir = "/test" if args.test else ""
print(args)

MODEL_DIR = './results/hjst{}/models/'.format(test_dir)
CONF_DIR = './configs{}'.format(test_dir)

layer_conf = LayerModelConfig()
layer_conf.link(os.path.join(CONF_DIR, 'layer.conf'), create_if_missing=True)
layer_conf.set_directory(MODEL_DIR, os.path.join(CONF_DIR, 'dataset.conf'))
layer_conf.update_file()

model = getattr(hjst_model.layers, args.model)
#kwargs = {k:v for k, v in map(lambda x: re.split(':', x), args.params)}
base_layer = args.base_layer
for e in list(sort_etypes(get_etypes()))[::-1]:
    if e.__name__ not in args.levels:
        continue
    base_layer = layer_conf.create_aglayer(e.__name__, model, base_layer, args.name)#, **kwargs)