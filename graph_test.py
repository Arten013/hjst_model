import itertools
import pickle
from copy import copy
import plyvel
import os
import MeCab
import re

from hjst_model.hierarchical_dataset import *
from hjst_model.hierarchical_model import *
from jstatutree.mltree.ml_lawdata import JStatutreeKVS
from jstatutree.xmltree import xml_lawdata
from jstatutree.myexceptions import *
from jstatutree.etypes import sort_etypes
from jstatutree.kvsdict import KVSDict
from gensim.models.doc2vec import TaggedDocument
from jstatutree.mltree import ml_etypes


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


class DatasetGensimGenerator(object):
    def __init__(self, kvsdict, preprocess):
        self.kvsdict = kvsdict
        self.preprocess = preprocess

    def __iter__(self):
        self.gen = (TaggedDocument(self.preprocess(s), [t]) for t, s in self.kvsdict.items())
        return self

    def __next__(self):
        return  next(self.gen)


BASEPATH = os.path.abspath(os.path.dirname(__file__))
REIKISET_PATH  = os.path.join(BASEPATH, "../reikiset/")
for i in range(47):
    t = time()
    register_directory(usr='neo4j', pw='pass', url='http://127.0.0.1:7474', levels=[xml_etypes.Law, xml_etypes.Article, xml_etypes.Sentence], basepath=os.path.join(REIKISET_PATH, '{0:02}'.format(i+1)))
    print(str(i+1), ':', time()-t)
    t = time()
