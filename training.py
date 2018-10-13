from hjst_model.config import LayerModelConfig
import os
import hjst_model.layers
import re
import argparse

parser = argparse.ArgumentParser(description='Training layer using kvs dataset.')
parser.add_argument('model',
                    help='training layer model')
parser.add_argument('dataset',
                    help='training set name')
parser.add_argument('--name', '-n', default='None',
                    help='model name')
parser.add_argument('--levels', '-l', nargs='*', default=None,
                    help='layer level')
parser.add_argument('--params', '-p', nargs='*',
        help='optional model parameter (format key:value)')
parser.add_argument('--force', default=False, action="store_true",
                    help='create model unless the same one has already existed.')
parser.add_argument('--test', default=False, action="store_true",
                    help='debug mode')
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
args_params = args.params or []
kwargs = {k:v for k, v in map(lambda x: re.split(':', x), args_params)}
if args.levels:
    levels = args.levels
else:
    dsc = layer_conf.dataset_config
    dsc.change_section(args.dataset)
    levels = dsc['levels']
for l in levels:
    layer_conf.create_layer(args.dataset, model, l.__name__, args.name, overwrite=args.force, **kwargs)
