from hjst_model.config import DatasetKVSConfig
import os
import argparse
import re

parser = argparse.ArgumentParser(description='Construct KVS dataset using reiki xml files.')
parser.add_argument('name',
                    help='dataset name (must be unique)')
parser.add_argument('--govcode', '-g', nargs='*', default='ALL',
                    help='an integer government code for the constructor')
parser.add_argument('--levels', '-l', nargs='*', default=['Law', 'Article', 'Sentence'],
                    help='layer level')
parser.add_argument('--maxsize', '-m', default='None',
                    help='an integer government code for the constructor')
parser.add_argument('--test', default=False, action="store_true",
                    help='debug mode')
args = parser.parse_args()

def decode_gc(gcs):
    if re.match("\d+$", gcs):
        return [int(gcs)]
    m = re.match("(\d+)-(\d+)$", gcs)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)


test_dir = "/test" if args.test else ""
DATASET_NAME = args.name
ROOT_CODE = [gc for gcl in args.govcode for gc in decode_gc(gcl)]
CONF_DIR = './configs{}'.format(test_dir)
REIKISET_DIR = '../home/reikiset'
KVS_DIR = './results/hjst{}/kvsdataset'.format(test_dir)
LEVELS = args.levels
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
    sect['gov_codes'] =  [gc for gcl in args.govcode for gc in decode_gc(gcl)]
    sect['levels'] = LEVELS
    sect['only_reiki'] = ONLY_REIKI
    sect['only_sentence'] = True
    sect['maxsize'] = args.maxsize
    dataset_conf.prepare_dataset()
