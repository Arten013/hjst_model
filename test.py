from xml_jstatutree.jstatutree.jstatute_dict import JStatutreeKVSDict, JSSentenceKVSDict
from xml_jstatutree import xml_lawdata
from xml_jstatutree.jstatutree.lawdata import SourceInterface
from xml_jstatutree.jstatutree.myexceptions import *
from xml_jstatutree.jstatutree.etypes import sort_etypes
from xml_jstatutree.jstatutree.kvsdict import KVSDict, KVSPrefixDict

import plyvel
import os
from concurrent.futures import ThreadPoolExecutor
import itertools
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle
from copy import copy

def find_all_files(directory, extentions=None):
    for root, dirs, files in os.walk(directory):
        if extentions is None or os.path.splitext(root)[1] in extentions:
            yield root
        for file in files:
            if extentions is None or os.path.splitext(file)[1] in extentions:
                yield os.path.join(root, file)

import MeCab
import re

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


class ModelLayerBase(object):
    def __init__(self, level, savepath):
        self.level = level
        self.savepath = savepath
        self.model = None

    def save(self):
        with open(self.savepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def train(self, dataset):
        pass

    def compare(self, elem1, elem2, *args, **kwargs):
        pass

class Doc2VecLayer(ModelLayerBase):
    def train(self, dataset):
        self.model = Doc2Vec(
            documents = list(dataset.iter_gensim_tagged_documents(self.level)),
            dm = 1,
            vector_size=300,
            window=8,
            min_count=10,
            workers=4
            )

    def compare(self, elem1, elem2, *args, **kwargs):
        return self.model.docvecs.similarity(elem1, elem2)

class ModelLayerBase(object):
    def __init__(self, level, savepath):
        self.level = level
        self.savepath = savepath
        self.model = None

    def save(self):
        with open(self.savepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def train(self, dataset):
        pass

    def compare(self, elem1, elem2, *args, **kwargs):
        pass

import editdistance
editdistance.eval('banana', 'bahama')
class LevenLayer(ModelLayerBase):
    def save(self):
        pass

    @classmethod
    def load(cls, path):
        raise Exception()

    def train(self, dataset):
        self.dataset = dataset

    def compare(self, elem1, elem2, *args, **kwargs):
        s1 = self.dataset.sentence_dict[self.level][elem1]
        s2 = self.dataset.sentence_dict[self.level][elem2]
        ret = 1 - editdistance.eval(s1, s2)/max(len(s1),len(s2))
        return ret


"""
import subprocess
class SimLevenLayer(ModelLayerBase):
    def save(self):
        pass

    @classmethod
    def load(cls, path):
        if 

    def train(self, dataset):
        sdir = os.path.split(self.savepath)[0]
        tmpfile_path = os.path.join(sdir, "dataset.txt")
        with open(tmpfile_path, "w") as f:
            f.write("\n".join(dataset.sentence_dict[self.level].values()))
        subprocess.check_call(
            "simstring -bu --database={savepath} < {tmpfile_path} && rm -f {tmpfile_path}".format(
                savepath=self.savepath,
                tmpfile_path=tmpfile_path
                ),
            shell=True)

    def compare(self, elem1, elem2, threshold, dataset):
        query = dataset.sentence_dict[self.level][elem1]
        target = dataset.sentence_dict[self.level][elem2]
        simstring_out = subprocess.call(
            "simstring --database={savepath} -t {threshold} -u < {query}".format(
                savepath=self.savepath,
                threshold=threshold,
                query=query
                )
            )
        for s in re.split("\s+", simstring_out):
            if s
        return self.model.docvecs.similarity(elem1, elem2)
"""
class ScoredPairKVS(KVSDict):
    DEFAULT_DBNAME = "ScoredPair.ldb"
    ENCODING = "utf8"
    PREFIX = "example-"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_dicts = dict()

    def add_by_iterable(self, list):
        self.add_by_iterable_pairs(itertools.combinations(iterable, 2))

    def add_by_iterable_pairs(self, iterable):
        for i1, i2 in iterable:
            self.add_scored_pair(i1, i2, 0)

    def add_scored_pair(self, i1, i2, score):
        i1, i2 = sorted([i1, i2])
        if not i1 in self.prefix_dicts:
            self.prefix_dicts[i1] = KVSPrefixDict(self.db, prefix={"{}-".format(i1)})
        self.prefix_dicts[i1][i2] = score

    def __iter__(self):
        for formar in self.prefix_dicts.keys():
            for lattar, score in self.prefix_dicts[formar].items():
                yield formar, latter, score

    def del_pair(self, i1, i2):
        i1, i2 = sorted([i1, i2])
        del self.prefix_dicts[i1][i2]

class LeveledScoredPairKVS(object):
    def __init__(self, path, levels):
        self.levels = levels
        self.level_dict = {
            l:ScoredPairKVS(path=os.path.join(path, "ScoredPairs-{}.ldb").format(l.__name__))
            for l in self.levels
            }

    def __getitem__(self, key):
        return self.level_dict[key]


class HierarchicalModel(object):
    def __init__(self, trainingset):
        self.trainingset = trainingset
        self.levels = trainingset.levels
        self.models = dict()
        self.thresholds = dict()
        self.top_rates = dict()

    def set_model(self, level, model_cls, savepath, threshold=0.3, top_rate=0.7):
        assert level in self.levels, "This level is not in levels"+str(levels)
        try:
            self.models[level] = model_cls.load(savepath)
            print("Load model:", savepath)
        except:
            self.models[level] = model_cls(level, savepath)
            print("Create model.")
        self.thresholds[level] = threshold
        self.top_rates[level] = top_rate

    def batch_training(self):
        assert len(self.models) == len(self.levels), "You must register models for all levels."
        for level in self.levels:
            if self.models[level].model is not None:
                continue
            self.models[level].train(self.trainingset)
            self.models[level].save()

    
    def refine_pairs(self, candidate_pairs):
        for li, level in enumerate(self.levels):
            t1 = time()
            candidate_num = len(candidate_pairs)
            scored_pairs = [None]*candidate_num
            si = 0
            for k1, k2 in candidate_pairs:                 
                score = self.models[level].compare(k1, k2)
                if self.thresholds[level] is None or score >= self.thresholds[level]:
                    scored_pairs[si] = (k1, k2, score)
                    si += 1
            topn = int(candidate_num * self.top_rates[level])
            print("threshold pass: {0:,}/{2:,}, top_rate pass: {1:,}/{2:,}".format(si, topn, candidate_num))
            if si < topn:
                scored_pairs = scored_pairs[:si]
            else:
                scored_pairs = sorted(scored_pairs[:si], key=lambda x: -x[2])[:topn]
            t2=time()
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            if level == self.levels[-1]:
                return scored_pairs
            candidate_pairs = list(
                itertools.chain.from_iterable(
                        [
                            itertools.product(
                                    testset.statutree_dict[k1],
                                    testset.statutree_dict[k2] 
                                )
                            for k1, k2, _ in scored_pairs
                        ] 
                    )
                )
            print("calculating next pairs:", time()-t2, "sec")
        
        raise("Unexpected error")

    def refine_pairs(self, candidate_pairs):
        for li, level in enumerate(self.levels):
            candidate_num = len(candidate_pairs[level])
            si = 0
            t1 = time()
            for k1, k2 in candidate_pairs[level]:                 
                score = self.models[level].compare(k1, k2)
                if self.thresholds[level] is None or score >= self.thresholds[level]:
                    candidate_pairs[level].add_scored_pair(k1, k2, score)
                    si += 1
                else:
                    candidate_pairs[level].del_pair(k1, k2)
            print("threshold pass: {0:,}/{2:,}".format(si, candidate_num))
            t2=time()
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            if level == self.levels[-1]:
                return scored_pairs
            candidate_pairs[level].add_by_iterable_pairs(
                itertools.chain.from_iterable(
                        [
                            itertools.product(
                                    testset.statutree_dict[k1],
                                    testset.statutree_dict[k2] 
                                )
                            for k1, k2, _ in scored_pairs
                        ] 
                    )
                )
            print("calculating next pairs:", time()-t2, "sec")
        
        raise("Unexpected error")

    def get_pairs(self, testset, dbpath):
        candidate_pairs = LeveledScoredPairKVS(path=dbpath, levels=self.levels)
        candidate_pairs.add_by_iterable(itertools.combinations(testset.sentence_dict[self.levels[0]].keys(), 2))
        return self.refine_pairs(candidate_pairs)



if __name__ == "__main__":
    from xml_jstatutree.xml_etypes import Law, Article, ArticleCaption, Paragraph, Sentence
    from time import time

    BASEPATH = os.path.abspath(os.path.dirname(__file__))
    RESULTPATH = os.path.join(BASEPATH, "aichi-aichi_pref")
    REIKISET_PATH  = os.path.join(BASEPATH, "../reikiset/")
    LEVELS = [Law, Article, Sentence]

    def setup_dataset(source_path, dbpath, levels):
        print("setup", dbpath)
        dataset = HierarchicalDataset(dbpath=dbpath, levels=levels, only_reiki=True, only_sentence=True)
        t = time()
        dataset.register(source_path)
        """
        print("reg time:", time() - t)
        print("len treedict", len(dataset.statutree_dict))
        for l in dataset.levels:
            print("len sentencedict({})".format(l.__name__), len(dataset.sentence_dict[l]))
        """
        return dataset

    TESTSET_PATH = os.path.join(REIKISET_PATH, "23/230006")
    TESTSET_DBPATH = os.path.join(BASEPATH, "aichi_pref_all.ldb")
    TRAININGSET_PATH = os.path.join(REIKISET_PATH, "23")
    TRAININGSET_DBPATH = os.path.join(BASEPATH, "aichi_all.ldb")

    testset = setup_dataset(TESTSET_PATH, TESTSET_DBPATH, LEVELS)
    trainingset = setup_dataset(TRAININGSET_PATH, TRAININGSET_DBPATH, LEVELS)

    """
    t = time()
    dataset.register(PATH)
    print("reg time:", time() - t)
    t=time()
    list(dataset.statutree_dict.items())
    print("load all tree time:", time() - t)
    t=time()
    for v in dataset.sentence_dict.values():
        list(v.items())
    print("load all sentence time:", time() - t)
    """
    """
    from pprint import pprint
    keys = ["23/230006/1847"]
    for level in dataset.levels:
        next_keys = []
        for key in keys:
            n = dataset.statutree_dict[key]
            for k in n:
                pprint(k)
                print(dataset.sentence_dict[level][k])
            next_keys.extend(n)
        keys = next_keys
    exit()
    """
        #for i, (k, s) in enumerate(dataset.sentence_dict[l].items()):
        #    print(k)
        #    print(s)
        #    if i > 20:
        #        break
    

    testpairs = list(itertools.combinations(testset.sentence_dict[Law].keys(), 2))[:100]
    testpairskvs = LeveledScoredPairKVS(RESULTPATH, LEVELS)
    testpairskvs[LEVELS[0]].add_by_iterable_pairs(testpairs)

    hmodels = HierarchicalModel(trainingset)
    hmodels.set_model(Law, Doc2VecLayer, os.path.join(BASEPATH, "all_LawD2V"), threshold=0.3, top_rate=1.0)
    hmodels.set_model(Article, Doc2VecLayer, os.path.join(BASEPATH, "all_ArticleD2V"), threshold=0.4, top_rate=1.0)
    hmodels.set_model(Sentence, LevenLayer, os.path.join(BASEPATH, "all_SentenceD2V"), threshold=0.5, top_rate=1.0)
    hmodels.batch_training()

    os.makedirs(RESULTPATH, exist_ok=True)
    with open(os.path.join(BASEPATH, "result_multi.csv"), "w") as f:
        sl = []
        sl.append("label1,sentence1,label2,sentence2,score")
        for k1, k2, score in hmodels.refine_pairs(testpairskvs):
            #print("score:", score)
            #print("s1:", k1)
            s1 = testset.sentence_dict[Sentence][k1]
            #print(s1)
            #print("s2:", k2)
            s2 = testset.sentence_dict[Sentence][k2]
            #print()
            sl.append(",".join([k1,s1,k2,s2,str(score)]))
        f.write("\n".join(sl))

    hmodels = HierarchicalModel(trainingset)
    hmodels.set_model(Law, Doc2VecLayer, os.path.join(BASEPATH, "all_LawD2V"), threshold=None, top_rate=1.0)
    hmodels.set_model(Article, Doc2VecLayer, os.path.join(BASEPATH, "all_ArticleD2V"), threshold=None, top_rate=1.0)
    hmodels.set_model(Sentence, LevenLayer, os.path.join(BASEPATH, "all_SentenceD2V"), threshold=0.5, top_rate=1.0)
    hmodels.batch_training()
    with open(os.path.join(BASEPATH, "result_single.csv"), "w") as f:
        sl = []
        sl.append("label1,sentence1,label2,sentence2,score")
        for k1, k2, score in hmodels.refine_pairs(testpairskvs):
            #print("score:", score)
            #print("s1:", k1)
            s1 = testset.sentence_dict[Sentence][k1]
            #print(s1)
            #print("s2:", k2)
            s2 = testset.sentence_dict[Sentence][k2]
            #print()
            sl.append(",".join([k1,s1,k2,s2,str(score)]))
        f.write("\n".join(sl))

