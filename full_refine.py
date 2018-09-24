from hjst_model.config import HierarchicalModelConfig, DatasetKVSConfig
import itertools
import os
from hjst_model.scored_pair import LeveledScoredPairKVS
import numpy as np
import re
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

parser = argparse.ArgumentParser(description='Training layer using kvs dataset.')
parser.add_argument('hmodel',
                    help='model name')
parser.add_argument('testset', default='None',
                    help='model name')
parser.add_argument('--test', default=False, action="store_true",
                    help='debug mode')
parser.add_argument('--workers', default=cpu_count()-2,
                    help='number of workers')
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
all_targets = list(testset.kvsdicts['edges']['Statutory'].keys())
all_pairs = list(itertools.combinations(all_targets, 2))
all_matrixes = dict()
dirpath = os.path.join('./results/hjst/layer_compare/{}/{}/'.format(HMODEL_NAME, TESTSET))
print(len(all_targets))
for l in all_targets:
    id_vec, mat = hmodel.get_layered_law_matrix(testset, l)
    path = os.path.join(dirpath, "matrixes", re.sub('/', '_', l))
    if os.path.exists(os.path.join(path, 'matrix.npy')):
        #print("load matrix")
        all_matrixes[l] = np.load(os.path.join(path, 'matrix.npy'))
        continue
    #print("get matrix")
    os.makedirs(path, exist_ok=True)
    id_vec, mat = hmodel.get_layered_law_matrix(testset, l)
    np.save(os.path.join(path, 'id_vector.npy'), id_vec)
    np.save(os.path.join(path, 'matrix.npy'), mat)
    all_matrixes[l] = mat

def layer_compare(lawpair):
    l1, l2 = lawpair
    path = os.path.join(dirpath, "compare", re.sub('/', '_', l1), re.sub('/', '_', l2))
    if os.path.exists(os.path.join(path, 'layered_matrix.npy')):
        return None, None
    os.makedirs(path, exist_ok=True)
    res_matrix = np.array([all_matrixes[l1][i] @ all_matrixes[l2][i].T for i in range(len(hmodel.levels))])
    np.save(os.path.join(path, 'layered_matrix.npy'), res_matrix)
    return l1, l2

def comp_lmat(path, lm1, lm2):
    return path, np.array([m1 @ lm2[i].T for i, m1 in enumerate(lm1)])
from time import time
t = time()
"""
with ProcessPoolExecutor() as ppe:
    futures = list()
    for l1, l2 in all_pairs:
        path = os.path.join(dirpath, "compare", re.sub('/', '_', l1), re.sub('/', '_', l2))
        if os.path.exists(os.path.join(path, 'layered_matrix.npy')):
            continue
        os.makedirs(path, exist_ok=True)
        futures.append(ppe.submit(comp_lmat, path, all_matrixes[l1], all_matrixes[l2]))
    for f in as_completed(futures):
        p, m = f.result()
        np.save(p, m)
        print(p)
"""
with ProcessPoolExecutor(args.workers) as ppe:
    for l1, l2 in ppe.map(layer_compare, all_pairs):
        print(l1, l2)

print("time:", time()-t)
