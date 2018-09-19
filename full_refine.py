from hjst_model.config import HierarchicalModelConfig, DatasetKVSConfig
import itertools
import os
from hjst_model.scored_pair import LeveledScoredPairKVS
import numpy as np
import re
import argparse
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser(description='Training layer using kvs dataset.')
parser.add_argument('hmodel',
                    help='model name')
parser.add_argument('testset', default='None',
                    help='model name')
parser.add_argument('--test', default=False, action="store_true",
                    help='debug mode')
args = parser.parse_args()


test_dir = "/test" if args.test else ""
print(args)

MODEL_DIR = './results/hjst{}/models/'.format(test_dir)
CONF_DIR = './configs{}'.format(test_dir)
TESTSET_DIR = './configs{}/dataset.conf'.format(test_dir)
HMODEL_NAME = args.hmodel
TESTSET = args.testset


model_conf = HierarchicalModelConfig()
model_conf.link(os.path.join(CONF_DIR, HMODEL_NAME+'.conf'), create_if_missing=True)
testset_conf = DatasetKVSConfig()
testset_conf.link(TESTSET_DIR)
testset_conf.change_section(TESTSET)
testset = testset_conf.prepare_dataset()
#scored_pairs = list(itertools.combinations(list(testset["texts"][LEVELS[0]].keys()), 2))
#scored_pairs_kvs = LeveledScoredPairKVS(os.path.join('./results/hjst/layer_compare', HMODEL_NAME, TESTSET), LEVELS)

hmodel = model_conf.model_generate()
#scored_pairs_kvs[LEVELS[0]].add_by_iterable_pairs(scored_pairs)
# hmodel.refine_pairs(testset, scored_pairs_kvs)
# hmodel = model_conf.model_generate()

def layer_compare(lawpair):
    l1, l2 = lawpair
    path = os.path.join('./results/hjst/layer_compare/{}/{}/npy/'.format(HMODEL_NAME, TESTSET), re.sub('/', '_', l1), re.sub('/', '_', l2))
    if os.path.exists(os.path.join(path, 'layered_matrix.npy')):
        return None, None
    os.makedirs(path, exist_ok=True)
    res_matrix = hmodel.get_layered_compare(testset, l1, l2)
    np.save(os.path.join(path, 'layered_matrix.npy'), res_matrix)
    return l1, l2
with ProcessPoolExecutor(10) as ppe:
    for l1, l2 in ppe.map(layer_compare, list(itertools.combinations(testset.kvsdicts['edges']['Statutory'].keys(), 2))):
        # sprint(l1, l2)
        pass


