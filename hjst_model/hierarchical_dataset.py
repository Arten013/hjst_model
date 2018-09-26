from jstatutree.mltree.ml_lawdata import JStatutreeKVS
from jstatutree.mltree import ml_etypes
from jstatutree.xmltree import xml_lawdata, xml_etypes
from jstatutree.graphtree import graph_etypes, graph_lawdata
from jstatutree.myexceptions import *
from jstatutree.etypes import sort_etypes
from jstatutree.kvsdict import KVSDict
from gensim.models.doc2vec import TaggedDocument

import multiprocessing
from time import time
import traceback
import concurrent
import plyvel
import os
import MeCab
import re
from pathlib import Path

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
        self.dataset_name = dataset_name
        self.kvsdicts['lawcodes'] = KVSDict(path=os.path.join(dbpath, 'lawcodes'))
        self.kvsdicts["texts"] = {
            l.__name__: KVSDict(path=os.path.join(dbpath, dataset_name, "texts", l.__name__)) for l in levels
            }
        self.kvsdicts["edges"] = {
            l.__name__: KVSDict(path=os.path.join(dbpath, dataset_name, "edges", l.__name__)) for l in levels[:-1]
            }
        self.kvsdicts['edges']['Statutory'] = KVSDict(path=os.path.join(dbpath, dataset_name, "edges", 'Statuotory'))
        self.morph_separator = Morph()

        
    def get_tags(self, roots, level):
        for root in roots:
            current_level = re.split('\(', os.path.split(root)[1])[0]
            if level == current_level:
                yield root
            else:
                yield from self.get_tags(self.kvsdicts['edges'][current_level][root], level) 
    
    def get_elem(self, code):
        parts = Path(code).parts

        query_root = self.kvsdicts["root"]['/'.join(parts[:3])]
        query_etype = getattr(ml_etypes, re.split('\(', parts[-1])[0])
        for e in query_root.depth_first_iteration(target_etypes=[query_etype]):
            if e.code == code:
                return e
        return None
       
    def close(self):
        self.kvsdicts['lawcodes'].close()
        if 'texts' in self.kvsdicts:
            for k in list(self.kvsdicts["texts"].keys()):
                self.kvsdicts["texts"][k].close()
                del self.kvsdicts["texts"][k]
            del self.kvsdicts["texts"]

        if 'edges' in self.kvsdicts:
            for k in list(self.kvsdicts["edges"].keys()):
                self.kvsdicts["edges"][k].close()
                del self.kvsdicts["edges"][k]
            del self.kvsdicts["edges"]
        super().close()

    def iter_lawcodes(self):
        yield from self.kvsdicts['lawcodes'][self.dataset_name]

    def __len__(self):
        return len(self.kvsdicts['lawcodes'][self.dataset_name])

    def set_data(self, reader):
        if self.only_reiki and not reader.lawdata.is_reiki():
            return False
        code = reader.lawdata.code
        if not self.kvsdicts['lawdata'].get(code, None):
            statutree = ml_etypes.convert_recursively(reader.get_tree())
            self.kvsdicts["lawdata"][code] = reader.lawdata
            self.kvsdicts["root"][code] = statutree
            for e in statutree.depth_first_iteration():
                if len(e.text) > 0:
                    self.kvsdicts["sentence"][code] = e.text
        else:
            statutree = self.kvsdicts['root'][code]
        writers = {
                l:self.kvsdicts["texts"][l.__name__].write_batch(transaction=True)
                for l in self.levels
            }
        for li, level in enumerate(self.levels):
            next_elems = list(statutree.depth_first_search(level, valid_vnode=True))
            if li == 0:
                self.kvsdicts['edges']['Statutory'][reader.lawdata.code] = [e.code for e in next_elems]
            for elem in next_elems:
                #print("reg:", "vnode" if elem.is_vnode else "node", elem.code)
                if self.only_sentence:
                    writers[level][elem.code] = "".join(elem.iter_sentences()) 
                else:
                    writers[level][elem.code] = "".join(elem.iter_texts())
                if level != self.levels[-1]:
                    self.kvsdicts["edges"][level.__name__][elem.code] = [e.code for e in elem.depth_first_search(self.levels[li+1], valid_vnode=True)]
        for writer in writers.values():
            writer.write()
        return True

    def register_directory(self, basepath, overwrite=False, maxsize=None):
        # if not overwrite and not self.kvsdicts["lawdata"].is_empty():
        #     print("skip registering", basepath)
        #     return
        exist_lawcodes = self.kvsdicts['lawcodes'].get(self.dataset_name, [])
        for path in find_all_files(basepath, [".xml"]):
            if maxsize is not None and len(exist_lawcodes) >= maxsize:
                print(exist_lawcodes)
                print(len(exist_lawcodes))
                break
            try:
                rr = xml_lawdata.ReikiXMLReader(path)
                rr.open()
                if rr.is_closed():
                    continue
                if rr.lawdata.code in exist_lawcodes:
                    continue
                if self.set_data(rr):
                    exist_lawcodes.append(rr.lawdata.code)
                rr.close()
            except LawError as e:
                pass
                #print(e)
        self.is_empty = False
        self.kvsdicts['lawcodes'][self.dataset_name] = exist_lawcodes

    def set_iterator_mode(self, level, tag=None, sentence=None, tokenizer='mecab', gensim=False):
        self.itermode_level = level
        self.itermode_tag = tag if tag is not None else self.__dict__.get('itermode_tag', True)
        self.itermode_sentence = sentence if sentence is not None else self.__dict__.get('itermode_sentence', True)
        self.tokenizer = self.morph_separator.surfaces if tokenizer == 'mecab' else tokenizer
        self.gensim = gensim

    def __iter__(self):
        assert self.__dict__.get('itermode_level', False), 'You must call set_iterator_mode before call iterator'
        if self.gensim:
            self.generator = map(lambda x: TaggedDocument(self.preprocess(x[1]), [x[0]]), self.kvsdicts['texts'][self.itermode_level].items())
        elif self.itermode_tag and self.itermode_sentence:
            self.generator = map(lambda x: (x[0], self.preprocess(x[1])), self.kvsdicts['texts'][self.itermode_level].items())
        elif self.itermode_sentence:
            self.generator = map(lambda x: self.preprocess(x), self.kvsdicts['texts'][self.itermode_level].values())
        elif self.itermode_tag:
            self.generator = self.kvsdicts['texts'][self.itermode_level].keys()
        else:
            self.generator = (x for x in [])
        return self

    def __next__(self):
        return next(self.generator)
    def iter_tagged_sentence(self, level):
        yield from ((t, self.preprocess(s)) for t, s in self.kvsdicts["texts"][level].items())

    def iter_gensim_tagged_documents(self, level, _tokenizer='mecab'):
        return DatasetGensimGenerator(kvsdict=self.kvsdicts['texts'][level], preprocess=self.preprocess)

    def preprocess(self, sentence):
        return self.tokenizer(sentence)

class HierarchicalGraphDataset(object):
    def __init__(self, loginkey, dataset_name, levels='ALL', only_reiki=True, only_sentence=False, *args, **kwargs):
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.gdb = graph_lawdata.ReikiGDB(**loginkey)
        self.morph_separator = Morph()
        self.levels = levels
        self.name= dataset_name
        with self.gdb.driver.session() as session:
            self.root_codes = [v[0] for v in session.run("MATCH (n:%s) RETURN n.code" % self.name).values()]

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
        with self.gdb.driver.session() as session:
            if len(govcode) == 2:
                res = session.run("""
                        MATCH (pref:Prefectures{code: '%s'})-[:PREF_OF]->(muni:Municipalities)
                        SET muni:%s
                        RETURN muni.code
                        """ % (govcode, self.name)
                        ).values()
                self.root_codes.extend([i[0] for i in res])
            else:
                session.run("""
                        MATCH (muni:Municipalities{code: '%s'})
                        SET muni:%s
                        """ % (govcode, self.name)
                        )
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
            with self.gdb.driver.session() as session:
                ret = [{'id': i[0], 'fulltext': i[1]} for i in session.run("""MATCH (muni:Municipalities{code: '%s'})-[]->(:Statutories)-[*1..]->(elem:%s) RETURN elem.id, elem.fulltext""" % (code, level_label))]
            return ret
        node_gen = (enode for enode_gen in (get_element_nodes(code) for code in self.root_codes) for enode in enode_gen)
        
        if self.itermode_tag and self.itermode_sentence:
            if self.itermode_gensim:
                self.generator = map(lambda x: TaggedDocument(tags=[x['id']], words=self.preprocess(x['fulltext'])), node_gen)
            else:
                self.generator = map(lambda x: (x['id'], self.preprocess(x['fulltext'])), node_gen)
        elif self.itermode_sentence:
            self.generator = map(lambda x: self.preprocess(x['fulltext']), node_gen)
        elif self.itermode_tag:
            self.generator = map(lambda x: x['id'], node_gen)
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

