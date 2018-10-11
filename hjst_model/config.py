from abconfig.abconfig import ABConfig, NEOConfig
import multiprocessing
import os
from .hierarchical_dataset import HierarchicalDataset, HierarchicalGraphDataset
from jstatutree.graphtree import graph_etypes, graph_lawdata
from jstatutree.etypes import get_etypes
from .hierarchical_model import HierarchicalModel
import re
"""
experiment = HJSTExperiment()

if os.exists(path):
    return

with experiment.create(path) as e:
    e.layers.set(Law, Doc2VecLayer, 0.3, None)
    e.layers.set(Article, Doc2VecLayer, 0.5, None)
    e.layers.set(Sentence, WVAverageLayer, 0.7, None)

    e.trainingset.set(loginkey, "LAS_aichi")

    e.train()

    e.testset("LAS_nagoya")
    e.refine()
"""

from . import layers as layer_module
def _encode_layer_class(layer):
    if isinstance(layer, str):
        assert layer in dir(layer_module), 'Invalid layer module "%s"' % layer
        return layer
    elif isinstance(layer, type):
        return _encode_layer_class(layer.__name__)
    TypeError('You must give layer by class or string.')

class HierarchicalModelConfig(ABConfig):
    CONF_ENCODERS = {
            'class': _encode_layer_class,
            'threshold': lambda x: str(round(x, 2))
            }
    CONF_DECODERS = {
            'class': lambda x: getattr(layer_module, x),
            'threshold': float
            }
    DEFAULT_SECTION = 'COMMON'

    @property
    def layer_config(self):
        if '_layer_config' not in self.__dict__:
            self._layer_config = LayerModelConfig()
            self._layer_config.link(self['layer_conf_path'], create_if_missing=False)
        return self._layer_config

    def set_directory(self, layer_conf_path):
        self['layer_conf_path'] = os.path.abspath(layer_conf_path)

    def set_layer_method(self, level, layer_class, threshold):
        level = level if isinstance(level, str) else level.__name__
        if self.has_section(level):
            print("This model already have", level, 'layer.')
        with self.batch_update(level) as layer_setting:
            layer_setting['type'] = 'method'
            layer_setting['class'] = layer
            layer_setting['threshold'] = threshold

    def set_layer_model(self, level, layer, threshold):
        level = level if isinstance(level, str) else level.__name__
        if self.has_section(level):
            print("This model already have", level, 'layer.')
        with self.batch_update(level) as layer_setting:
            layer_setting['type'] = 'model'
            layer_setting['name'] = layer
            layer_setting['threshold'] = threshold

    def model_generate(self):
        hmodel = HierarchicalModel()
        for level, layer in self.iter_sections():
            level = getattr(graph_etypes, level)
            if layer['type'] == 'method':
                raise 'Not Implemented yet'
            else:
                hmodel.set_trained_model_layer(level.__name__, self.layer_config.load_model(self['name']), layer['threshold'])
        return hmodel

class LayerModelConfig(ABConfig):
    DEFAULT_SECTION = 'COMMON'
    CONF_ENCODERS = {
            'model': _encode_layer_class,
            'levels': lambda l: ','.join([x if isinstance(x, str) else x.__name__ for x in l]),
            }
    CONF_DECODERS = {
            'model': lambda x: getattr(layer_module, x),
            'levels': lambda l: [getattr(graph_etypes, x) for x in l.split(',')],
            }

    @property
    def dataset_config(self):
        if '_dataset_config' not in self.__dict__:
            self._dataset_config = DatasetKVSConfig()
            self._dataset_config.link(self['dataset_conf_path'], create_if_missing=False)
        return self._dataset_config

    def set_directory(self, model_dir, dataset_conf_path):
        self['model_dir'] = os.path.abspath(model_dir)
        self['dataset_conf_path'] = os.path.abspath(dataset_conf_path)

    def get_model_name(self, trainingset_name=None, model_class=None, level=None):
        trainingset_name = trainingset_name if trainingset_name is not None else self['trainingset']
        model_class = model_class if model_class is not None else self['model']
        level = level if level is not None else self['level']
        return '{trainingset_name}_{model}-{level}'.format(
                trainingset_name=trainingset_name,
                model=model_class.__name__,
                level=level if isinstance(level, str) else level.__name__
            )

    def load_model(self, name=None):
        name = name or self.section_name
        with self.temporal_section_change(name) as s:
            return s['model'].load(s.model_path)

    def create_aglayer(self, level, model_class, base_layer, model_name = None, **kwargs):
        name = model_name or self.get_model_name(trainingset_name, model_class)
        name = name+"-"+level
        if self.has_section(name):
            print('Layer', name, 'has already exists.')
            return
        with self.temporal_section_change(base_layer):
            trainingset = self["trainingset"]
        with self.batch_update(name) as sect:
            sect['model'] = model_class
            sect['base_layer'] = base_layer
            sect['trainingset'] = trainingset
            sect['level'] = level
            for k, v in kwargs.items():
                sect[k] = v
            model = self['model'](self['level'], self.model_path)
            with self.dataset_config.temporal_section_change(trainingset):
                ds = self.dataset_config.prepare_dataset()
                model.train(ds, self.load_model(base_layer))
                model.save()
                ds.close()
                del ds
        return name        
        
    def create_layer(self, trainingset_name, model_class, level, model_name = None, **kwargs):
        name = model_name or self.get_model_name(trainingset_name, model_class)
        name = name+"-"+level
        if self.has_section(name):
            print('Layer', name, 'has already exists.')
            return
        with self.batch_update(name) as sect:
            sect['model'] = model_class
            sect['trainingset'] = trainingset_name
            sect['level'] = level
            for k, v in kwargs.items():
                sect[k] = v
            model = self['model'](self['level'], self.model_path)
            self.dataset_config.change_section(trainingset_name)
            ds = self.dataset_config.prepare_dataset(registering=False)
            model.train(ds, **kwargs)
            model.save()
            ds.close()
            del ds
        return model

    @property
    def model_path(self):
        return os.path.join(self['model_dir'], re.sub("-", "/", self.section_name) + '.lmodel')

def _gov_codes_encoder(codes):
    if len(codes) == 0:
        return 'None'
    if 'ALL' in codes or codes == ['{0:02}'.format(v) for v in range(1,47)]:
        return 'ALL'
    codes = sorted(map(int, list(set(codes))))
    ret = []
    for code in codes:
        if code < 100:
            ret.append('{0:02}'.format(code))
        else:
            if code//10000 in codes:
                continue
            ret.append('{0:06}'.format(code))
    return ', '.join(ret)

class DatasetConfigBase(ABConfig):
    CONF_ENCODERS = {
            'levels': lambda l: ','.join([x if isinstance(x, str) else x.__name__ for x in l]),
            'gov_codes': _gov_codes_encoder,
            'keywords': lambda x: ','.join(x) if x is not None else None
            }
    CONF_DECODERS = {
            'levels': lambda l: [getattr(graph_etypes, x) for x in l.split(',')],
            'only_sentence': lambda x: x == 'True',
            'only_reiki': lambda x: x == 'True',
            'gov_codes': lambda x: ['{0:02}'.format(v) for v in range(1,47)] if x == 'ALL' else ([] if x == 'None' else re.split(', ', x)),
            'maxsize': lambda x: None if x == 'None' else int(x),
            'keywords': lambda x: re.split(',', x)
            }
    DEFAULT_SECTION = 'COMMON'

    def set_directory(self, dataset_basepath):
        self['dataset_basepath'] = os.path.abspath(dataset_basepath)
        self['gov_codes'] = []

    def iter_dataset_paths(self):
        if self['gov_codes'] == '':
            yield self['dataset_basepath']
            raise StopIteration
        for gov_code in self['gov_codes']:
            gov_dir = gov_code if len(gov_code) == 2 else gov_code[:2]+'/'+gov_code
            yield os.path.join(self['dataset_basepath'], gov_dir)

class DatasetNEOConfig(NEOConfig, DatasetConfigBase):
    def prepare_dataset(self, graph_construction=True, registering=True, workers=multiprocessing.cpu_count()):
        assert self.section.name != self.__class__.DEFAULT_SECTION, 'You must set dataset before get hgd instance'
        dataset = HierarchicalGraphDataset.init_by_config(self)
        for dataset_path in self.iter_dataset_paths():
            if graph_construction:
                print('construct graph from:', self.dataset_path)
                graph_lawdata.register_directory(levels=self['levels'], basepath=dataset_path, loginkey=self.loginkey, workers=workers, only_reiki=self['only_reiki'], only_sentence=['only_sentence'])
            if registering:
                dataset.add_government(os.path.split(dataset_path)[0])
        return dataset

class DatasetKVSConfig(DatasetConfigBase):
    def set_directory(self, dataset_basepath, savedir):
        super().set_directory(dataset_basepath)
        self['savedir'] = savedir

    def prepare_dataset(self, registering=True):
        dataset = HierarchicalDataset(self['savedir'], self.section_name, self['levels'],self['only_reiki'], self['only_sentence'])
        if not registering:
            return dataset
        for path in self.iter_dataset_paths():
            print(path)
            dataset.register_directory(path, overwrite=True, maxsize=self.get('maxsize', None), keywords=self.get('keywords', None))
        additional_govs = self['gov_codes']
        with self.temporal_section_change(self.DEFAULT_SECTION):
            self['gov_codes'] = self['gov_codes'] + additional_govs
        return dataset

