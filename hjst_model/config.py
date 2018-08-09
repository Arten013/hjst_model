from abconfig.abconfig import ABConfig, NEOConfig
import multiprocessing
import os
from .hierarchical_dataset import HierarchicalGraphDataset
from jstatutree.graphtree import graph_etypes, graph_lawdata

def NEOConfigManager(object):
    def __init__(self, basepath):
        self.configs = {
                'model': HierarchicalModelConfig(os.path.join('model.conf')),
                'layer': LayerFrameworkConfig(os.path.join('layer.conf')),
                'dataset': DatasetNEOConfig(os.path.join('dataset.conf')),
                }

class HierarchicalModelConfig(ABConfig):
    CONF_ENCODERS = {}
    CONF_DECODERS = {
            'trained': lambda x: x == 'True',
            }
    def __init__(self, layer_config, dataset_config, path=None):
        self.layer_config = layer_config
        self.dataset_config = dataset_config
        self['layer_config'] = layer_config.path
        self['dataset_config'] = dataset_config.path

    def create(self, model_path, layerframework_name, trainingset_name, overwrite=False):
        model_name = os.path.splitext(os.path.split(model_path)[1])
        assert self.layer_config.has_section(layerframework_name), ''
        if not overwrite and (self.has_section(model_name) or os.path.exists(model_path)):
            return
        self.change_section(model_name, exists)
        self['model_path'] = model_path
        self['layerframework_name'] = model_name
        self['trainingset'] = trainingset
        self['trained'] = False

    def train(self):
        self['trained'] = True

class LayerFrameworkConfig(ABConfig):
    def set_layer(self, level, layer_cls, threshold=0.3, overwrite=False):
        level = level if isinstance(level, str) else level.__name__
        if self.has_section(name) and not overwrite:
            return
        self.change_section(level, create_if_missing=True)
        self['threshold'] = threshold
        self['layer'] = layer_cls.__name__

class DatasetNEOConfig(NEOConfig):
    CONF_ENCODERS = {
            'levels': lambda l: ','.join([x if isinstance(x, str) else x.__name__ for x in l]),
            }
    CONF_DECODERS = {
            'levels': lambda l: [getattr(graph_etypes, x) for x in l.split(',')],
            'only_sentence': lambda x: x == 'True',
            'only_reiki': lambda x: x == 'True',
            }

    def __init__(self, levels, dataset_basepath, result_basepath, path=None, only_reiki=True, only_sentence=True):
        super().__init__(path)
        self['levels'] = levels
        self['only_reiki'] = only_reiki
        self['only_sentence'] = only_sentence
        self['dataset_basepath'] = os.path.abspath(dataset_basepath)
        self['result_basepath'] = os.path.abspath(result_basepath)


    def add_dataset(self, name, root_code, exist_ok=True):
        assert not (exist_ok and self.parser.has_section(name)), 'Dataset {} has already existed.'.format(name)
        self.change_section(name, create_if_missing=True)
        self.section['root_code'] = root_code

    def set_dataset(self, name):
        assert self.parser.has_section(name), 'Dataset {} does not exist.'.format(name)
        self.change_section(name, create_if_missing=False)

    @property
    def dataset_path(self):
        root_code = self['root_code'] if len(self['root_code']) == 2 else self['root_code'][:2]+'/'+self['root_code']
        return os.path.join(self['dataset_basepath'], root_code)

    @property
    def result_path(self):
        return os.path.join(self['result_basepath'], self.section.name)

    def prepare_dataset(self, registering=True, workers=multiprocessing.cpu_count()):
        assert self.section.name != 'DEFAULT', 'You must set dataset before get hgd instance'
        dataset = HierarchicalGraphDataset.init_by_config(self)
        if registering:
            print('reg:', self.dataset_path)
            graph_lawdata.register_directory(levels=self['levels'], basepath=self.dataset_path, loginkey=self.loginkey, workers=workers, only_reiki=self['only_reiki'], only_sentence=['only_sentence'])
            dataset.add_government(self['root_code'])
        return dataset
