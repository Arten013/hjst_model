from jstatutree.mltree.ml_lawdata import JStatutreeKVS
from jstatutree.mltree import ml_etypes
from jstatutree.xmltree import xml_lawdata, xml_etypes
from jstatutree.graphtree import graph_etypes, graph_lawdata
from jstatutree.myexceptions import *
from jstatutree.etypes import sort_etypes
from jstatutree.kvsdict import KVSDict
from gensim.models.doc2vec import TaggedDocument

import multiprocessing
from neo4jrestclient.client import Node
from time import time
import traceback
import concurrent
import plyvel
import os
import MeCab
import re

def find_all_files(directory, extentions=None):
    for root, dirs, files in os.walk(directory):
        if extentions is None or os.path.splitext(root)[1] in extentions:
            yield root
        for file in files:
            if extentions is None or os.path.splitext(file)[1] in extentions:
                yield os.path.join(root, file)

class Morph(object):
    def __init__(self, tagging_mode = ' '):     
        self.tagger = MeCab.Tagger(tagging_mode)
        self.tagger.parse('')

    def iter_surface(self, text):        
        morph_list = self.tagger.parseToNode(text)
        while morph_list:
            yield morph_list.surface
            morph_list = morph_list.next

    def surfaces(self, text):
        return list(self.iter_surface(text))

class HierarchicalDataset(JStatutreeKVS):
    def __init__(self, dbpath, dataset_name, levels, only_reiki=True, only_sentence=False, *args, **kwargs):
        super().__init__(path=dbpath)
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.levels = sort_etypes(levels)
        self.kvsdicts["texts"] = {
            l: KVSDict(path=os.path.join(dbpath, dataset_name, "texts", l.__name__)) for l in levels
            }
        self.kvsdicts["edges"] = {
            l: KVSDict(path=os.path.join(dbpath, dataset_name, "edges", l.__name__)) for l in levels[:-1]
            }
        self.morph_separator = Morph()

    def close(self):
        for k in list(self.kvsdicts["texts"].keys()):
            self.kvsdicts["texts"][k].close()
            del self.kvsdicts["texts"][k]
        del self.kvsdicts["texts"]

        for k in list(self.kvsdicts["edges"].keys()):
            self.kvsdicts["edges"][k].close()
            del self.kvsdicts["edges"][k]
        del self.kvsdicts["edges"]
        super().close()

    def set_data(self, reader):
        if self.only_reiki and not reader.lawdata.is_reiki():
            return
        statutree = ml_etypes.convert_recursively(reader.get_tree())
        writers = {
                l:self.kvsdicts["texts"][l].write_batch(transaction=True)
                for l in self.levels
            }
        for li, level in enumerate(self.levels):
            for elem in statutree.depth_first_search(level, valid_vnode=True):
                #print("reg:", "vnode" if elem.is_vnode else "node", elem.code)
                if self.only_sentence:
                    writers[level][elem.code] = "".join(elem.iter_sentences()) 
                else:
                    writers[level][elem.code] = "".join(elem.iter_texts())
                if level != self.levels[-1]:
                    self.kvsdicts["edges"][level][elem.code] = [e.code for e in elem.depth_first_search(self.levels[li+1], valid_vnode=True)]
        code = reader.lawdata.code
        self.kvsdicts["lawdata"][code] = reader.lawdata
        self.kvsdicts["root"][code] = statutree
        for e in statutree.depth_first_iteration():
            if len(e.text) > 0:
                self.kvsdicts["sentence"][code] = e.text
        for writer in writers.values():
            writer.write()

    def register_directory(self, basepath, overwrite=False):
        if not overwrite and not self.kvsdicts["lawdata"].is_empty():
            print("skip registering")
            return
        for path in find_all_files(basepath, [".xml"]):
            try:
                rr = xml_lawdata.ReikiXMLReader(path)
                rr.open()
                if rr.is_closed():
                    continue
                self.set_data(rr)
                rr.close()
            except LawError as e:
                pass
                #print(e)
        self.is_empty = False

    def set_iterator_mode(self, level, tag=None, sentence=None):
        self.itermode_level = level
        self.itermode_tag = tag if tag is not None else self.__dict__.get('itermode_tag', True)
        self.itermode_sentence = sentence if sentence is not None else self.__dict__.get('itermode_sentence', True)

    def __iter__(self):
        assert self.__dict__.get('itermode_level', False), 'You must call set_iterator_mode before call iterator'
        if self.itermode_tag and self.itermode_sentence:
            self.generator = map(lambda x: (x[0], self.preprocess(x[1])), self.kvsdicts['texts'][self.itermode_level].items())
        elif self.itermode_sentence:
            self.generator = map(lambda x: self.preprocess(x), self.kvsdicts['texts'][self.itermode_level].values())
        elif self.itermode_tag:
            self.generator = self.kvsdicts.keys()
        else:
            self.generator = (x for x in [])
        return self

    def __next__(self):
        return next(self.generator)
    def iter_tagged_sentence(self, level):
        yield from ((t, self.preprocess(s)) for t, s in self.kvsdicts["texts"][level].items())

    def iter_gensim_tagged_documents(self, level):
        return DatasetGensimGenerator(kvsdict=self.kvsdicts['texts'][level], preprocess=self.preprocess)

    def preprocess(self, sentence):
        return self.morph_separator.surfaces(sentence)

class GraphDatasetConfig(graph_lawdata.DBConfig):
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
        self['levels'] = [l.capitalize() if isinstance(l, str) else l.__name__ for l in levels]
        self['only_reiki'] = only_reiki
        self['only_sentence'] = only_sentence
        self['dataset_basepath'] = os.path.abspath(dataset_basepath)
        self['result_basepath'] = os.path.abspath(result_basepath)

    def __getitem__(self, key):
        return self.CONF_DECODERS.get(key, str)(self.section[key])

    def __setitem__(self, key, value):
        self.section[key] = self.CONF_ENCODERS.get(key, str)(value)

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

class HierarchicalGraphDataset(object):
    def __init__(self, loginkey, dataset_name, levels='ALL', only_reiki=True, only_sentence=False, *args, **kwargs):
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.gdb = graph_lawdata.ReikiGDB(**loginkey)
        self.morph_separator = Morph()
        self.levels = levels
        self.name= dataset_name
        try:
            self.root_codes = [node['code'] for node in self.gdb.db.labels.get(self.name).all()]
        except KeyError:
            self.root_codes = []

    @classmethod
    def init_by_config(cls, config):
        return cls(
                    loginkey=config.loginkey,
                    dataset_name=config.section.name,
                    levels=config['levels'],
                    only_reiki=config['only_reiki'],
                    only_sentence=config['only_reiki']
                )

    def add_government(self, govcode):
        if len(govcode) == 2:
            res = self.gdb.db.query("""MATCH (pref:Prefectures{code: '%s'})-[:PREF_OF]->(muni:Municipalities) SET muni:%s RETURN muni.code""" % (govcode, self.name))
            self.root_codes.extend([i[0] for i in res])
        else:
            res = self.gdb.db.query("""MATCH (muni:Municipalities{code: '%s'}) SET muni:%s""" % (govcode, self.name))
            self.root_codes.append(govcode)
        self.root_codes = list(set(self.root_codes))

    def set_iterator_mode(self, level, tag=None, sentence=None, gensim=None):
        self.itermode_level = level
        self.itermode_tag = tag if tag is not None else self.__dict__.get('itermode_tag', None)
        self.itermode_sentence = sentence if sentence is not None else self.__dict__.get('itermode_sentence', None)
        self.itermode_gensim = gensim if gensim is not None else self.__dict__.get('itermode_gensim', None)

    def __iter__(self):
        assert self.__dict__.get('itermode_level', False), 'You must call set_iterator_mode before call iterator'
        def get_element_nodes(code):
            level_label = (self.itermode_level.capitalize() if isinstance(self.itermode_level, str) else self.itermode_level.__name__) + 's'
            ret = [{'fullname': i[0], 'fulltext': i[1]} for i in self.gdb.db.query("""MATCH (muni:Municipalities{code: '%s'})-[]->(:Statutories)-[*1..]->(elem:%s) RETURN elem.fullname, elem.fulltext""" % (code, level_label), returns=(str, str))]
            return ret
        node_gen = (enode for enode_gen in (get_element_nodes(code) for code in self.root_codes) for enode in enode_gen)
        
        if self.itermode_tag and self.itermode_sentence:
            if self.itermode_gensim:
                self.generator = map(lambda x: TaggedDocument(tags=[x['fullname']], words=self.preprocess(x['fulltext'])), node_gen)
            else:
                self.generator = map(lambda x: (x['fullname'], self.preprocess(x['fulltext'])), node_gen)
        elif self.itermode_sentence:
            self.generator = map(lambda x: self.preprocess(x['fulltext']), node_gen)
        elif self.itermode_tag:
            self.generator = map(lambda x: x['fullname'], node_gen)
        else:
            self.generator = (x for x in [])
        return self

    def __next__(self):
        return next(self.generator)

    def iter_tagged_sentence(self, level):
        self.set_iterator_mode(level, tag=True, sentence=True)
        return self

    def iter_gensim_tagged_documents(self, level):
        self.set_iterator_mode(level, tag=True, sentence=True, gensim=True)
        return self

    def preprocess(self, sentence):
        return self.morph_separator.surfaces(sentence)

class DatasetGensimGenerator(object):
    def __init__(self, kvsdict, preprocess):
        self.kvsdict = kvsdict
        self.preprocess = preprocess

    def __iter__(self):
        self.gen = (TaggedDocument(self.preprocess(s), [t]) for t, s in self.kvsdict.items())
        return self

    def __next__(self):
        return  next(self.gen)

