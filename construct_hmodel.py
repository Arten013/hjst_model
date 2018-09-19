from hjst_model.config import HierarchicalModelConfig, DatasetKVSConfig
import itertools
import os
from hjst_model.scored_pair import LeveledScoredPairKVS
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser(description='Training layer using kvs dataset.')
parser.add_argument('hmodel',
                    help='training layer model')
parser.add_argument('--levels', '-l', nargs='+', default=None,
                    help='layer level')
parser.add_argument('--layers', '-p', nargs='+',
        help='optional model parameter (format key:value)')
parser.add_argument('--test', default=False, action="store_true",
                    help='debug mode')
args = parser.parse_args()

test_dir = "/test" if args.test else ""
print(args)

MODEL_DIR = './results/hjst{}/models/'.format(test_dir)
CONF_DIR = './configs{}'.format(test_dir)
LEVELS = args.levels or ['Law', 'Article', 'Sentence']

model_conf = HierarchicalModelConfig()
model_conf.link(os.path.join(CONF_DIR, args.hmodel+'.conf'), create_if_missing=True)
if len(model_conf) == 0:
    model_conf.set_directory(os.path.join(CONF_DIR, 'layer.conf'))
    for level, layer in zip(LEVELS, args.layers):
        model_conf.set_layer_model(level, layer, 0)