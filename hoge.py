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

from jstatutree.graphtree import graph_etypes, graph_lawdata
import concurrent
class HierarchicalGraphDataset(object):
    def __init__(self, usr, pw, url, levels='ALL', only_reiki=True, only_sentence=False, *args, **kwargs):
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.gdb = graph_lawdata.ReikiGDB(usr=usr, pw=pw, url=url)
        self.morph_separator = Morph()
        self.levels = levels

    def set_data(self, path):
        reader =xml_lawdata.ReikiXMLReader(path)
        if self.gdb.load_lawdata(reader.lawdata.municipality_code, reader.lawdata.file_code) is not None:
            print('skip(exists): '+str(reader.lawdata.name) )
        reader.open()
        if reader.is_closed():
            return 'skip(cannot open): '+str(reader.lawdata.name)
        if not self.only_reiki or reader.lawdata.is_reiki():
            self.gdb.set_from_reader(reader, levels=self.levels)
            reader.close()
            return 'register: '+str(reader.lawdata.name)
        else:
            reader.close()
            return 'skip(not reiki): '+str(reader.lawdata.name)


from jstatutree.xmltree import xml_etypes
from time import time
def register_directory(usr, pw, url, levels, basepath, workers=3):
    def split_list(alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                for i in range(wanted_parts) ]
    path_lists = split_list(list(find_all_files(basepath, [".xml"])), workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as proc_exec:
        futures = [proc_exec.submit(register_from_pathlist, pathlist=path_lists[i], usr=usr, pw=pw, url=url, levels=levels) for i in range(workers)]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as exc:
                print('%s' % (exc,))

def register_from_pathlist(pathlist, *hgd_arg, **hgd_kwargs):
    dataset = HierarchicalGraphDataset(*hgd_arg, **hgd_kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(dataset.set_data, path) for path in pathlist]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as exc:
                print('%s' % (exc,))
import neo4jrestclient
BASEPATH = os.path.abspath(os.path.dirname(__file__))
REIKISET_PATH  = os.path.join(BASEPATH, "../reikiset/")
t = time()
dataset = HierarchicalGraphDataset(usr='neo4j', pw='pass', url='http://0.0.0.0:7474', levels=[xml_etypes.Law, xml_etypes.Article, xml_etypes.Sentence])
print(dataset.gdb.db.query("MATCH (n:Statutories) WHERE n.code='0006' RETURN n")[0])
