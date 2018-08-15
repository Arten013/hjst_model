import os
import itertools
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle
from copy import copy

from hjst_model.hierarchical_dataset import *
from hjst_model.hierarchical_model import *
from hjst_model.layers import *
from hjst_model.config import DatasetNEOConfig

def setup_dataset(source_path, dbpath, levels):
    print("setup", dbpath)
    dataset = HierarchicalGraphDataset(dbpath=dbpath, dataset_name="LAS", levels=levels, only_reiki=True, only_sentence=True)
    t = time()
    dataset.register_directory(source_path, overwrite=False)
    """
    print("reg time:", time() - t)
    print("len law", len(dataset["lawdata"]))
    for l in dataset.levels:
        print("len sentencedict({})".format(l.__name__), len(dataset["texts"][l]))
    """
    return dataset

if __name__ == "__main__":
    from jstatutree.graphtree.graph_etypes import *
    from time import time

    BASEPATH = os.path.abspath(os.path.dirname(__file__))
    RESULTBASEPATH = os.path.join(BASEPATH, 'results/hjst_model/test')
    REIKISET_PATH  = os.path.join(BASEPATH, "../jltset/")
    LEVELS = [Law, Article, Sentence]
    TRAININGSET_PATH = os.path.join(REIKISET_PATH, "23/230006")

    CONFPATH = './configs/testset.conf'
    conf = DatasetNEOConfig(levels=LEVELS, dataset_basepath=REIKISET_PATH, result_basepath=RESULTBASEPATH, path=CONFPATH)
    conf.set_logininfo(host='127.0.0.1')
    # conf.update()
    try:
        conf.set_dataset('nazo')
        trainingset = conf.prepare_dataset(registering=True, workers=2)
    except AssertionError:
        conf.add_dataset('nazo', '000000')
        trainingset = conf.prepare_dataset(workers=2)
    conf.update()
    hmodels = HierarchicalModel(trainingset)
    hmodels.set_layer(Law, Doc2VecLayer, os.path.join(RESULTBASEPATH, 'layers', "aichi_LawD2V.model"), threshold=0.3)
    hmodels.set_layer(Article, Doc2VecLayer, os.path.join(RESULTBASEPATH, 'layers', "aichi_ArticleD2V.model"), threshold=0.4)
    hmodels.set_layer(Sentence, WVAverageLayer, os.path.join(RESULTBASEPATH, 'layers', "aichi_SentenceWVA.model"), threshold=0.7)
    hmodels.batch_training()

