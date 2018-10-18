from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import re
import os
from gensim.models.doc2vec import TaggedDocument
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from copy import copy
import shutil
import numpy as np
import pickle
import multiprocessing
from jstatutree import kvsdict
from jstatutree.tree_element import TreeElement
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from glove import Glove as GloveModel
from glove import Corpus as GloveCorpus
from .tokenizer import load_spm, get_spm, Morph
from .hierarchical_dataset import ReikiKVSDict



class LayerBase(object):
    __slots__ = [
            'level',
            'savepath',
            'spm_path',
            'vecs',
            '_tokenizer',
            'tokenizer_type'
        ]
    
    def __init__(self, level, savepath, **kwargs):
        self.level = level
        self.savepath = savepath
        self.spm_path = os.path.join(self.savepath, '..', 'spm')
        self.init_vecs()
        self._tokenizer = None
        self.tokenizer_type = None

    def _calc_vec(self, text):
        raise "Not implemented"
        
    def init_vecs(self):
        vecspath = os.path.join(self.savepath, 'vecs.ldb')
        print('Load kvsdict from the path below:')
        print(vecspath)
        self.vecs = kvsdict.KVSDict(vecspath)
        
    @property
    def tokenizer(self):
        assert self.tokenizer_type, "You must first train model before call tokenizer"
        if self.tokenizer_type == "spm":
            self._tokenizer = self._tokenizer or load_spm(self.spm_path).EncodeAsPieces
        elif self.tokenizer_type == "mecab":
            self._tokenizer = self._tokenizer or Morph().surfaces
        else:
            raise "There is no tokenizer named "+str(self.tokenizer_type)
        return self._tokenizer
    
    def prepare_tokenizer(self, dataset, tokenizer_type, vocab_size):
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'spm':
            self._tokenizer = get_spm(dataset, self.spm_path, vocab_size).EncodeAsPieces
            
    def tokenize(self, text):
        return text if isinstance(text, list) else self.tokenizer(text)
    
    def get(self, key, text, add_if_missing=False):
        return self.vecs.get(key, self.add_vector_by_text(key, text))
            
    def add_vector_by_text(self, key, text):
        vec = self.vectorize_text(text)
        self[key] = vec
        return vec
            
    def vectorize_text(self, text):
        return self._calc_vec(self.tokenize(text))

    def save(self):
        self.vecs = None
        self._tokenizer = None
        with open(os.path.join(self.savepath, 'layer_instance.pkl'), "wb") as f:
            pickle.dump(self, f)
        self.init_vecs()

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'layer_instance.pkl'), "rb") as f:
            loaded = pickle.load(f)
        vardict = dict()
        for varname in cls.__slots__:
            if varname in ['vecs']:
                continue
            try:
                vardict[varname] = getattr(loaded, varname)
            except:
                print('WARNING: loaded instance has no attribute', varname, '.')
                print('The attribute skipped.')
        del loaded
        ret = cls(**vardict)
        for k, v in vardict.items():
            setattr(ret, k, v)
        return ret

    def compare(self, elem1, elem2):
        return np.dot(self.vecs[elem1], self.vecs[elem2])
    
    def train(self, dataset, tokenizer='mecab', vocab_size=8000, **kwargs):
        assert self.__class__ != LayerBase, 'You cannot use LayerBase class without override.'
        vocab_size = int(vocab_size)
        self.prepare_tokenizer(dataset, tokenizer, vocab_size)

    def __getitem__(self, key):
        return self.vecs[key]

    @classmethod
    def is_model(self):
        return True

    def __str__(self):
        return self.__class__.__name__
    
    def __del__(self):
        if hasattr(self, 'vecs'):
            del self.vecs
    
class AggregationLayerBase(LayerBase):
    def __init__(self, level, savepath):
        super().__init__(level, savepath)
        os.makedirs(self.savepath, exist_ok=True)
    
    def train(self, dataset, base_layer):
        with self.vecs.write_batch() as wb:
            for pnode, cnodes in dataset.kvsdicts["edges"][self.level].items():
                wb[pnode] = self._calc_vec(np.array([base_layer[n] for n in cnodes]))
                
    def _calc_vec(self, matrix):
        raise "You must implement aggregation function"
        
class AverageAGLayer(AggregationLayerBase):
    def _calc_vec(self, matrix):
        v = np.sum(matrix, axis=0)
        return v/np.linalg.norm(v)

class RandomAGLayer(AggregationLayerBase):
    def _calc_vec(self, matrix):
        print(matrix.shape)
        v = np.random.normal(loc=0, scale=1000, size=(matrix.shape[1],))
        return v/np.linalg.norm(v)

class MyGloVeModel(object):
    def fit(self, sentences, size, window, iter, workers=None, **kwargs):
        self.corpus = GloveCorpus()
        self.model = GloveModel(no_components=size, **kwargs) 
        self.corpus.fit(corpus=sentences, window=window, ignore_missing=True)
        self.model.fit(matrix=self.corpus.matrix, epochs=iter, no_threads=workers or multiprocessing.cpu_count(), verbose=True)
        self.wv = Word2VecKeyedVectors(size)
        for word, wid in self.corpus.dictionary.items():
            print("add", word, wid, self.model.word_vectors[wid][:5], "...")
            self.wv.add(word, self.model.word_vectors[wid], True)
    
    def save(self, savepath):
        os.makedirs(savepath, exist_ok=True)
        self.corpus.save(os.path.join(savepath, "corpus.model"))
        self.model.save(os.path.join(savepath, "model.model"))
        self.wv.save(os.path.join(savepath, "word_vectors.model"))
    
    @classmethod
    def load(cls, savepath):
        glove = cls()
        glove.corpus = GloveCorpus.load(os.path.join(savepath, "corpus.model"))
        glove.model = GloveModel.load(os.path.join(savepath, "model.model"))
        glove.wv = Word2VecKeyedVectors.load(os.path.join(savepath, "word_vectors.model"))
        return glove

def _get_wvmodel_class(wvmodel_name):
    try:
        return {
                "word2vec": Word2Vec,
                "fasttext": FastText,
                "glove": MyGloVeModel,
        }[wvmodel_name.lower()]
    except:
        raise Exception("There is no wvmodel named "+str(wvmodel_name))
        
class SWEMLayerBase(LayerBase):
    __slots__ = LayerBase.__slots__ + [
        'wvmodel_path',
        'wbmodel',
        'wbmodel_class'
    ]
    
    def __init__(self, level, savepath, **kwargs):
        super().__init__(level, savepath, **kwargs)
        self.wvmodel_path = os.path.join(self.savepath, '..', 'wvmodel')
        
    def train_wvmodel(self, dataset, vocab_size, wvmodel="word2vec",**kwargs):
        if self.wvmodel_class == Word2Vec:
            self.wvmodel = Word2Vec(
                    sentences = dataset,
                    size = kwargs.get("size") or 500,
                    window = kwargs.get("window") or 4,
                    min_count= kwargs.get("min_count") or 10,
                    iter =kwargs.get("iter") or 10,
                    max_vocab_size=vocab_size,
                    workers=multiprocessing.cpu_count(),
                    **kwargs
                )
        elif self.wvmodel_class == FastText:
            self.wvmodel = FastText(
                    sentences = dataset, 
                    size = kwargs.get("size") or 500,
                    window = kwargs.get("window") or 4,
                    min_count = kwargs.get("min_count") or 10,
                    iter =kwargs.get("iter") or 10,
                    max_vocab_size=vocab_size,
                    workers=multiprocessing.cpu_count(),
                    **kwargs
                )
        elif self.wvmodel_class == MyGloVeModel:
            self.wvmodel = MyGloVeModel()
            self.wvmodel.fit(
                    sentences = dataset, 
                    size = kwargs.get("size") or 500,
                    window = kwargs.get("window") or 4,
                    iter =kwargs.get("iter") or 10,
                    workers=multiprocessing.cpu_count(),
                    **kwargs
                )
        self.wvmodel.save(self.wvmodel_path)
    
    def train(self, dataset, tokenizer='mecab', vocab_size=8000, wvmodel="word2vec", **kwargs):
        vocab_size = int(vocab_size)
        super().train(dataset, tokenizer, vocab_size)
        self.wvmodel_class = _get_wvmodel_class(wvmodel)
        if os.path.exists(self.wvmodel_path):
            self.load_wvmodel()
        else:
            text_unit = kwargs.get("text_unit") or dataset.levels[0]
            text_unit = text_unit.__name__ if issubclass(text_unit, TreeElement) else text_unit
            dataset.set_iterator_mode(text_unit, tag=False, sentence=True, tokenizer=self.tokenizer)
            self.train_wvmodel(dataset, vocab_size=vocab_size, **kwargs)
        dataset.set_iterator_mode(self.level, tag=True, sentence=True, tokenizer=self.tokenizer)
        with self.vecs.write_batch() as wb:
            for tag, text in dataset:
                if not isinstance(tag, str):
                    print('WARNING: text tag is not str (type {})'.format(type(tag)))
                    tag = str(tag)
                wb[tag] = self._calc_vec(text)

    def save(self):
        tmp_wvmodel = self.wvmodel
        self.wvmodel = None
        super().save()
        self.wvmodel = tmp_wvmodel
    
    def load_wvmodel(self):
        self.wvmodel = self.wvmodel_class.load(self.wvmodel_path)

class SWEMAverageLayer(SWEMLayerBase):        
    def _calc_vec(self, text):
        arr = np.array([self.wvmodel.wv[v] for v in text if v in self.wvmodel.wv])
        if arr.shape == np.array([]).shape:
            print("WARNING: a zero vector allocated for the text below:")
            print(text)
            return np.zeros(self.wvmodel.wv.vector_size)
        v = np.sum(arr, axis=0)
        return v/np.linalg.norm(v)
    
class SWEMMaxLayer(SWEMLayerBase):
    def _calc_vec(self, text):
        arr = np.array([self.wvmodel.wv[v] for v in text if v in self.wvmodel.wv])
        if arr.shape == np.array([]).shape:
            print("WARNING: a zero vector allocated for the text below:")
            print(text)
            return np.zeros(self.wvmodel.wv.vector_size)
        v = np.max(arr, axis=0)
        return v/np.linalg.norm(v)

class SWEMConcatLayer(SWEMLayerBase):
    def _calc_vec(self, text):
        arr = np.array([self.wvmodel.wv[v] for v in text if v in self.wvmodel.wv])
        if arr.shape == np.array([]).shape:
            print("WARNING: a zero vector allocated for the text below:")
            print(text)
            return np.zeros(self.wvmodel.wv.vector_size*2)
        v1, v2 = np.sum(arr, axis=0), np.max(arr, axis=0)
        v = np.concatenate([v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)], axis=None)
        return v/np.linalg.norm(v)
    
class WMDLayer(SWEMLayerBase):    
    @property
    def docs(self):
        return self.vecs
    
    def _calc_vec(self, text):
        return text
    
    def get(self, key, text, add_if_missing=False):
        raise 'WMDLayer do not have document vectors.'
            
    def add_vector_by_text(self, key, text):
        raise 'WMDLayer do not have document vectors.'
            
    def vectorize_text(self, text):
        raise 'WMDLayer do not have document vectors.'
        
    def add_texts(self, *texts):
        with self.vecs.write_batch() as wb:
            for tag, text in texts:
                if not isinstance(tag, str):
                    print('WARNING: text tag is not str (type {})'.format(type(tag)))
                    tag = str(tag)
                wb[tag] = self._calc_vec(text)
    
    def compare_by_texts(self, text1, text2):
        return self.wvmodel.wv.wmdistance(self.tokenize(text1),  self.tokenize(text2))

    def save(self):
        self._tokenizer = None
        with open(os.path.join(self.savepath, 'layer_instance.pkl'), "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        layer = super().load(cls, path)
        layer.load_wvmodel()
        return layer

    def compare(self, elem1, elem2):
        return self.wvmodel.wv.wmdistance(self.vecs[elem1], self.vecs[elem2])

    def __getitem__(self, key):
        raise 'WMDLayer do not have document vectors.'

class RandomLayer(LayerBase):
    def train(self, dataset, scale=1, size=500):
        dataset.set_iterator_mode(self.level, tag=True, sentence=False)
        with self.vecs.write_batch() as wb:
            for tag in list(dataset):
                #print(tag)
                v = np.random.normal(loc=0, scale=float(scale), size=(int(size),))
                wb[tag] = v/np.linalg.norm(v)
    
class Doc2VecLayer(LayerBase):
    def train(self, dataset, tokenizer='mecab', vocab_size=8000, **kwargs):
        super().train(dataset, tokenizer, vocab_size)
        dataset.set_iterator_mode(self.level, gensim=True, tokenizer=self.tokenizer)
        print('train doc2vec model')
        dvmodel = Doc2Vec(
            documents = dataset,
            dm = 1,
            vector_size=300,
            sample=0.000001,
            negative=5,
            window=5,
            min_count=1,
            epochs=10,
            workers=multiprocessing.cpu_count()
            )
        print('save doc2vec model')
        dvmodel.save(os.path.join(self.savepath, 'doc2vec.model'))
        dataset.set_iterator_mode(self.level, tag=True, sentence=False)
        print('set vectors to kvsdict')
        with self.vecs.write_batch() as wb:
            for tag in list(dataset):
                wb[tag] = dvmodel.docvecs[tag]
        print('training complete')

    @classmethod
    def load_dvmodel(cls, path):
        return Doc2Vec.load(os.path.join(path, 'doc2vec.model'))

class TfidfLayer(LayerBase):
    __slots__ = LayerBase.__slots__ + [
        'tag_idx_dict',
        'transformer',
        'matrix'
    ]
            
    def _calc_vec(self, text):
        return self.transformer.transform(text)

    def train(self, dataset, tokenizer='mecab', vocab_size=8000, idf=True, lsi_size=None):
        super().train(dataset, tokenizer, vocab_size)
        vocab_size = int(vocab_size)
        idf = bool(idf)
        lsi_size = int(lsi_size) if lsi_size else None
        dataset.set_iterator_mode(self.level, tag=True, sentence=False, tokenizer=self.tokenizer)
        self.tag_idx_dict = {k:i for i, k in enumerate(dataset)}
        dataset.set_iterator_mode(self.level, tag=False, sentence=True, tokenizer=self.tokenizer)
        count_vectorizer = CountVectorizer(
            input='content',
            #max_df=0.5, 
            #min_df=1, 
            lowercase = False,
            max_features=vocab_size
            )
        steps = [('CountVectorize', count_vectorizer)]
        if idf:
            steps.append(("TfidfTransform", TfidfTransformer()))
        if lsi_size:
            steps.append(
                        ( 
                            "TruncatedSVD",
                            TruncatedSVD(n_components=lsi_size, algorithm='randomized', n_iter=10, random_state=42)
                        )
            )
        self.transformer = Pipeline(steps)
        self.matrix = self.transformer.fit_transform([" ".join(s) for s in dataset])
        dataset.set_iterator_mode(self.level, tag=True, sentence=False)
        with self.vecs.write_batch() as wb:
            for tag in list(dataset):
                wb[tag] = self.matrix[self.tag_idx_dict[tag]]

    def save(self):
        os.makedirs(self.savepath, exist_ok=True)
        tmp_mat = self.matrix
        with open(os.path.join(self.savepath, 'matrix.npy'), "wb") as f:
            np.save(f, self.matrix)
            del self.matrix
        super().save()
        self.matrix = tmp_mat
    
    def matrix_idx(self, tag):
        return self.tag_idx_dict[key]
    
    def load_matrix(self, matrix):
        with open(os.path.join(path, 'matrix.npy'), "rb") as f:
            return np.load(f)
