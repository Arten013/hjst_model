from xml_jstatutree.jstatutree.jstatute_dict import JStatutreeKVSDict, JSSentenceKVSDict
from xml_jstatutree import xml_lawdata
from xml_jstatutree.jstatutree.lawdata import SourceInterface
from xml_jstatutree.jstatutree.myexceptions import *
from xml_jstatutree.jstatutree.etypes import sort_etypes

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

class HierarchicalDataset(object):
    def __init__(self, dbpath, levels, only_reiki=True, only_sentence=False, *args, **kwargs):
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.levels = sort_etypes(levels)
        if os.path.exists(dbpath):
            self.db = plyvel.DB(dbpath, create_if_missing=False)
            self.is_empty = False
        else:
            self.db = plyvel.DB(dbpath, create_if_missing=True)
            self.is_empty = True
        self.sentence_dict = {
            l:JSSentenceKVSDict(self.db, only_reiki=only_reiki, level=l) for l in levels
            }
        self.statutree_dict = JStatutreeKVSDict.init_as_prefixed_db(self.db, only_reiki=only_reiki, levels=self.levels)
        self.morph_separator = Morph()

    def set_data(self, reader):
        assert issubclass(reader.__class__, SourceInterface)
        statutree = reader.get_tree()
        key = reader.lawdata.code
        if not self.only_reiki or reader.lawdata.is_reiki():
            self.statutree_dict[key] = statutree
            for level in self.levels:
                for elem in statutree.depth_first_search(level):
                    if elem.etype == level:
                        key = elem.code
                    else:
                        virtual_elem_key = elem.code+"/{}(1)".format(level.__name__)
                        key = virtual_elem_key
                    if self.only_sentence:
                        self.sentence_dict[level][key] = "".join(elem.iter_sentences()) 
                    else:
                        self.sentence_dict[level][key] = "".join(elem.iter_texts())

    def register(self, basepath, overwrite=False):
        if not overwrite and not self.is_empty:
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
        yield from ((t, self.preprocess(s)) for t, s in self.sentence_dict[level].items())

    def iter_gensim_tagged_documents(self, level):
        yield from (TaggedDocument(self.preprocess(s), [t]) for t, s in self.sentence_dict[level].items())

    def preprocess(self, sentence):
        return self.morph_separator.surfaces(sentence)