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

from hierarchical_dataset import *
from hierarchical_model import *
from model_layers import *
from scored_pair import *

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

