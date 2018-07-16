import os
import itertools
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle
from copy import copy

from hjst_model.hierarchical_dataset import *
from hjst_model.hierarchical_model import *
from hjst_model.layers import *

def setup_dataset(source_path, dbpath, levels):
    print("setup", dbpath)
    dataset = HierarchicalDataset(dbpath=dbpath, dataset_name="LAS", levels=levels, only_reiki=True, only_sentence=True)
    t = time()
    dataset.register_directory(source_path, overwrite=False)
    print("reg time:", time() - t)
    print("len law", len(dataset["lawdata"]))
    for l in dataset.levels:
        print("len sentencedict({})".format(l.__name__), len(dataset["texts"][l]))
    return dataset

if __name__ == "__main__":
    from jstatutree.mltree.ml_etypes import *
    from time import time

    BASEPATH = os.path.abspath(os.path.dirname(__file__))
    RESULTBASEPATH = os.path.join(BASEPATH, 'results/hjst_model')
    RESULTPATH = os.path.join(RESULTBASEPATH, "all-aichi_pref")
    REIKISET_PATH  = os.path.join(BASEPATH, "../reikiset/23/230006")
    LEVELS = [Law, Article, Sentence]

    TRAININGSET_PATH = os.path.join(REIKISET_PATH, "")
    TRAININGSET_DBPATH = os.path.join(RESULTBASEPATH, 'dataset', "japan_all.ldb")
    trainingset = setup_dataset(TRAININGSET_PATH, TRAININGSET_DBPATH, LEVELS)

    hmodels = HierarchicalModel(trainingset)
    hmodels.set_layer(Law, Doc2VecLayer, os.path.join(RESULTBASEPATH, 'layers', "all_LawD2V.model"), threshold=0.3)
    hmodels.set_layer(Article, Doc2VecLayer, os.path.join(RESULTBASEPATH, 'layers', "all_ArticleD2V.model"), threshold=0.4)
    hmodels.set_layer(Sentence, Doc2VecLayer, os.path.join(RESULTBASEPATH, 'layers', "all_SentenceD2V.model"), threshold=0.7)
    hmodels.batch_training()

