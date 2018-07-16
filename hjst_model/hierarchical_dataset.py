from jstatutree.mltree.ml_lawdata import JStatutreeKVS
from jstatutree.xmltree import xml_lawdata
from jstatutree.myexceptions import *
from jstatutree.etypes import sort_etypes
from jstatutree.kvsdict import KVSDict
from gensim.models.doc2vec import TaggedDocument
from jstatutree.mltree import ml_etypes

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
        self.morph_list = self.tagger.parseToNode(text)
        while self.morph_list:
            yield self.morph_list.surface
            self.morph_list = self.morph_list.next

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

    def iter_tagged_sentence(self, level):
        yield from ((t, self.preprocess(s)) for t, s in self.self.kvsdicts["texts"][level].items())

    def iter_gensim_tagged_documents(self, level):
        yield from (TaggedDocument(self.preprocess(s), [t]) for t, s in self.kvsdicts["texts"][level].items())

    def preprocess(self, sentence):
        return self.morph_separator.surfaces(sentence)