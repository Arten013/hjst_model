from hjst_model.config import HierarchicalModelConfig, DatasetKVSConfig
import itertools
import os
from hjst_model.scored_pair import LeveledScoredPairKVS

MODEL_DIR = './results/hjst/models/'
DATASET_CONFIG = './'
HMODEL_NAME = 'ddw_tokai_refinement_am500'
CONF_DIR = './configs'
LEVELS = ['Law', 'Article', 'Sentence']
TESTSET = 'am500'
import re

model_conf = HierarchicalModelConfig()
model_conf.link(os.path.join(CONF_DIR, HMODEL_NAME+'.conf'), create_if_missing=True)
testset_conf = DatasetKVSConfig()
testset_conf.link('./configs/dataset.conf')
testset_conf.change_section(TESTSET)
testset = testset_conf.prepare_dataset()
scored_pairs = list(itertools.combinations(list(testset["texts"][LEVELS[0]].keys()), 2))
scored_pairs_kvs = LeveledScoredPairKVS(os.path.join('./results/hjst/scored_pairs/', HMODEL_NAME, TESTSET), LEVELS)

if len(model_conf) == 0:
    model_conf.set_directory(os.path.join(CONF_DIR, 'layer.conf'))
    model_conf.set_layer_model('Law', 'tokai-Doc2VecLayer-Law', 0.5)
    model_conf.set_layer_model('Article', 'tokai-Doc2VecLayer-Article', 0.7)
    model_conf.set_layer_model('Sentence', 'tokai-WVAverageLayer-Sentence', 0.8)

    hmodel = model_conf.model_generate()
    scored_pairs_kvs[LEVELS[0]].add_by_iterable_pairs(scored_pairs)
    hmodel.refine_pairs(testset, scored_pairs_kvs)
else:
    hmodel = model_conf.model_generate()
for l1, l2 in itertools.combinations(testset.iter_lawcodes(), 2):
    path = os.path.join('./results/hjst/scored_pairs/{}/{}/csvs/'.format(HMODEL_NAME, TESTSET), re.sub('/', '_', l1), re.sub('/', '_', l2))
    scored_pairs_kvs.get_score_csv(path, testset, hmodel, l1, l2)
