from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from copy import copy

import pickle
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