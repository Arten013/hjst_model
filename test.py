import os
import itertools
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument import pickle
from copy import copy

from hierarchical_dataset import *
from hierarchical_model import *
from layers import *
from scored_pair import *

if __name__ == "__main__":
    from jstatutree.mltree.ml_etypes import Law, Article, ArticleCaption, Paragraph, Sentence
    from time import time

    BASEPATH = os.path.abspath(os.path.dirname(__file__))
    RESULTPATH = os.path.join(BASEPATH, "all-aichi_pref")
    REIKISET_PATH  = os.path.join(BASEPATH, "../reikiset/")
    LEVELS = [Law, Article, Sentence]

    def setup_dataset(source_path, dbpath, levels):
        print("setup", dbpath)
        dataset = HierarchicalDataset(dbpath=dbpath, dataset_name="LAS", levels=levels, only_reiki=True, only_sentence=True)
        t = time()
        dataset.register_directory(source_path, overwrite=False)
        """
        print("reg time:", time() - t)
        print("len law", len(dataset["lawdata"]))
        for l in dataset.levels:
            print("len sentencedict({})".format(l.__name__), len(dataset["texts"][l]))
        """
        return dataset

    TESTSET_PATH = os.path.join(REIKISET_PATH, "23/230006")
    TESTSET_DBPATH = os.path.join(BASEPATH, "aichi_pref_all.ldb")
    TRAININGSET_PATH = os.path.join(REIKISET_PATH, "")
    TRAININGSET_DBPATH = os.path.join(BASEPATH, "japan_all.ldb")
    testset = setup_dataset(TESTSET_PATH, TESTSET_DBPATH, LEVELS)
    trainingset = setup_dataset(TRAININGSET_PATH, TRAININGSET_DBPATH, LEVELS)
    """
    sentences = [
        "{rank},{count}".format(rank=i+1, count=count, sentence=sentence)
        for i, count  in enumerate(sorted(KVSValuesCounter(trainingset.sentence_dict[Sentence]).values(), key=lambda x: -x))
        ]

    with open("sentence_count_jreiki_all_onlydata.csv", "w") as f:
        f.write("\n".join(sentences))

    with open("sentence_count_jreiki_100000.csv", "w") as f:
        f.write("\n".join(sentences[:100000]))

    with open("sentence_count_jreiki_10000.csv", "w") as f:
        f.write("\n".join(sentences[:10000]))

    with open("sentence_count_jreiki_1000.csv", "w") as f:
        f.write("\n".join(sentences[:1000]))
    """
    #exit()

    for i in [100, 200, 300, 400, 500 , 600, 700]:
        SAMPLE_NUM = i
        testpairs = list(itertools.combinations(list(testset["texts"][Law].keys())[:SAMPLE_NUM], 2))
        testpairskvs = LeveledScoredPairKVS(os.path.join(RESULTPATH, "DDD{}".format(SAMPLE_NUM)), LEVELS)
        testpairskvs[LEVELS[0]].add_by_iterable_pairs(testpairs)

        hmodels = HierarchicalModel(trainingset)
        hmodels.set_layer(Law, Doc2VecLayer, os.path.join(BASEPATH, "all_LawD2V.model"), threshold=0.3)
        hmodels.set_layer(Article, Doc2VecLayer, os.path.join(BASEPATH, "all_ArticleD2V.model"), threshold=0.4)
        hmodels.set_layer(Sentence, Doc2VecLayer, os.path.join(BASEPATH, "all_SentenceD2V.model"), threshold=0.7)
        hmodels.batch_training()
        hmodels.refine_pairs(testset, testpairskvs)

        os.makedirs(RESULTPATH, exist_ok=True)

        testpairskvs = LeveledScoredPairKVS(os.path.join(RESULTPATH, "__D{}".format(SAMPLE_NUM)), LEVELS)
        testpairskvs[LEVELS[0]].add_by_iterable_pairs(testpairs)

        hmodels = HierarchicalModel(trainingset)
        hmodels.set_layer(Law, Doc2VecLayer, os.path.join(BASEPATH, "all_LawD2V.model"), threshold=None)
        hmodels.set_layer(Article, Doc2VecLayer, os.path.join(BASEPATH, "all_ArticleD2V.model"), threshold=None)
        hmodels.set_layer(Sentence, Doc2VecLayer, os.path.join(BASEPATH, "all_SentenceD2V.model"), threshold=0.7)
        hmodels.batch_training()
        hmodels.refine_pairs(testset, testpairskvs)

