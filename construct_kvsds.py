from hjst_model.config import DatasetKVSConfig
import os
import argparse

parser = argparse.ArgumentParser(description='Construct KVS dataset using reiki xml files.')
parser.add_argument('name',
                    help='dataset name (must be unique)')
parser.add_argument('--govcode', '-g', type=int, nargs='*', default='ALL',
                    help='an integer government code for the constructor')
parser.add_argument('--level', '-l', nargs='*', default=['Law', 'Article', 'Sentence'],
                    help='layer level')
parser.add_argument('--maxsize', '-m', type=int, default='None',
                    help='an integer government code for the constructor')
args = parser.parse_args()



DATASET_NAME = args.name
ROOT_CODE = args.govcode
CONF_DIR = './configs'
REIKISET_DIR = '../reikiset'
KVS_DIR = './results/hjst/kvsdataset'
LEVELS = args.level
ONLY_REIKI = True
ONLY_SENTENCE = True

dataset_conf = DatasetKVSConfig()
try:
    dataset_conf.link(os.path.join(CONF_DIR, 'dataset.conf'), create_if_missing=False)
except:
    dataset_conf.link(os.path.join(CONF_DIR, 'dataset.conf'), create_if_missing=True)
    dataset_conf.set_directory(REIKISET_DIR, KVS_DIR)
    dataset_conf.update_file()
# if dataset_conf.has_section(DATASET_NAME):
#     print('Dataset', DATASET_NAME, 'has already exists.')
#     exit()
with dataset_conf.batch_update(DATASET_NAME) as sect:
    sect['gov_codes'] = args.govcode
    sect['levels'] = LEVELS
    sect['only_reiki'] = ONLY_REIKI
    sect['only_sentence'] = True
    sect['maxsize'] = args.maxsize
    dataset_conf.prepare_dataset()
