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
from layers import *
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

    testpairs = list(itertools.combinations(list(testset.sentence_dict[Law].keys())[:400], 2))
    testpairskvs = LeveledScoredPairKVS(os.path.join(RESULTPATH, "DDL400"), LEVELS)
    testpairskvs[LEVELS[0]].add_by_iterable_pairs(testpairs)

    hmodels = HierarchicalModel(trainingset)
    hmodels.set_layer(Law, Doc2VecLayer, os.path.join(BASEPATH, "aichi_LawD2V.model"), threshold=0.3)
    hmodels.set_layer(Article, Doc2VecLayer, os.path.join(BASEPATH, "aichi_ArticleD2V.model"), threshold=0.4)
    hmodels.set_layer(Sentence, LevenLayer, os.path.join(BASEPATH, "aichi_SentenceD2V.model"), threshold=0.5)
    hmodels.batch_training()

    os.makedirs(RESULTPATH, exist_ok=True)
    with open(os.path.join(BASEPATH, "result_multi.csv"), "w") as f:
        sl = []
        sl.append("label1,sentence1,label2,sentence2,score")
        for k1, k2, score in hmodels.refine_pairs(trainingset, testpairskvs):
            #print("score:", score)
            #print("s1:", k1)
            s1 = testset.sentence_dict[Sentence][k1]
            #print(s1)
            #print("s2:", k2)
            s2 = testset.sentence_dict[Sentence][k2]
            #print()
            sl.append(",".join([k1,s1,k2,s2,str(score)]))
        f.write("\n".join(sl))

    testpairskvs = LeveledScoredPairKVS(os.path.join(RESULTPATH, "__L400"), LEVELS)
    testpairskvs[LEVELS[0]].add_by_iterable_pairs(testpairs)

    hmodels = HierarchicalModel(trainingset)
    hmodels.set_layer(Law, Doc2VecLayer, os.path.join(BASEPATH, "aichi_LawD2V.model"), threshold=None)
    hmodels.set_layer(Article, Doc2VecLayer, os.path.join(BASEPATH, "aichi_ArticleD2V.model"), threshold=None)
    hmodels.set_layer(Sentence, LevenLayer, os.path.join(BASEPATH, "aichi_SentenceD2V.model"), threshold=0.5)
    hmodels.batch_training()
    with open(os.path.join(BASEPATH, "result_single.csv"), "w") as f:
        sl = []
        sl.append("label1,sentence1,label2,sentence2,score")
        for k1, k2, score in hmodels.refine_pairs(testset, testpairskvs):
            #print("score:", score)
            #print("s1:", k1)
            s1 = testset.sentence_dict[Sentence][k1]
            #print(s1)
            #print("s2:", k2)
            s2 = testset.sentence_dict[Sentence][k2]
            #print()
            sl.append(",".join([k1,s1,k2,s2,str(score)]))
        f.write("\n".join(sl))

