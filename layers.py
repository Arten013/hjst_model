from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from copy import copy
import numpy as np
import pickle
import editdistance

class LayerBase(object):
    def refine_scored_pairs(self, scored_pairs, threshold):
        for k1, k2, _ in scored_pairs:                 
            score = self.compare(k1, k2)
            scored_pairs.add_scored_pair(k1, k2, score)

class MethodLayerBase(LayerBase):
    def __init__(self, dataset, level):
        self.dataset = dataset
        self.level = level

    def compare(self, elem1, elem2):
        pass

    @classmethod
    def is_model(self):
        return False

class ModelLayerBase(LayerBase):
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

    def compare(self, elem1, elem2):
        pass

    @classmethod
    def is_model(self):
        return True

    def __str__(self):
        return self.__class__.__name__

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

    def compare(self, elem1, elem2):
        return self.model.docvecs.similarity(elem1, elem2)

    def __str__(self):
        return "Doc2Vec model"

class TfidfLayer(ModelLayerBase):
    def train(self, dataset):
        if self.vectorizer is not None:
            return None
        self.vectorizer = TfidfVectorizer(
            input='content',
            max_df=0.5, 
            min_df=1, 
            max_features=30000, 
            norm='l2'
            )

        self.vectorizer.fit_transform(self._iter_dataset(dataset.iter_tagged_documents(self.level)))

    def _iter_dataset(self, iterable):
        for tag, data in iterable:
            pass



    def compare(self, elem1, elem2):
        return self.model.docvecs.similarity(elem1, elem2)

class LevenLayer(MethodLayerBase):
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