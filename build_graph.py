from hjst_model.config import DatasetNEOConfig
import os

DATASET_NAME = 'aichi_muni'
ROOT_CODE = 230006
CONF_DIR = './configs'
REIKISET_DIR = '../reikiset'
RESULT_DIR = './results/hjst'
LEVELS = ['Law', 'Article', 'Sentence']
ONLY_REIKI = True
LOGININFO = {
        'host':'localhost',
        'port':'17474',
        'boltport':'7687',
        'usr':'neo4j',
        'pw':'pass'
        }
WORKERS = 4

dataset_conf = DatasetNEOConfig()
try:
    dataset_conf.link(os.path.join(CONF_DIR, 'dataset.conf'), create_if_missing=False)
except:
    dataset_conf.link(os.path.join(CONF_DIR, 'dataset.conf'), create_if_missing=True)
    dataset_conf.set_directory(REIKISET_DIR)
    dataset_conf.set_logininfo(**LOGININFO)
    dataset_conf.update_file()
if dataset_conf.has_section(DATASET_NAME):
    print('Dataset', DATASET_NAME, 'has already exists.')
    exit()
with dataset_conf.batch_update(DATASET_NAME) as sect:
    sect['root_code'] = ROOT_CODE
    sect['levels'] = LEVELS
    sect['only_reiki'] = ONLY_REIKI
    sect['only_sentence'] = True # not implemented yet
    dataset_conf.prepare_dataset(registering=True, graph_construction=True,workers=WORKERS)
