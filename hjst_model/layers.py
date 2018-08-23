from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import re
import os
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from copy import copy
import numpy as np
import pickle
import editdistance
import multiprocessing
from neo4j.v1 import GraphDatabase
from jstatutree import kvsdict
try:
    import sentencepiece as spm
except:
    print('WARNING: you cannot sentencepiece model')



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
        os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
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

def get_spm(dataset, level, savepath, vocab_size=8000):
    model_path = os.path.join(savepath, 'model.model')
    corpus_path = os.path.join(savepath, 'corpus.txt')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        dataset.set_iterator_mode(level, tag=False, sentence=True, tokenizer=lambda x: x)
        with open(corpus_path, 'w') as f:
            f.write('\n'.join(dataset))
        spm.SentencePieceTrainer.Train('--input={} --model_prefix=model --vocab_size={}'.format(corpus_path, vocab_size))
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

class WVAverageModel(object):
    def __init__(self, wvmodel, savepath):
        self.wvmodel = wvmodel
        self.savepath = savepath
        os.makedirs(self.savepath, exist_ok=True)
        self.init_vecs()

    def init_vecs(self):
        self.vecs = kvsdict.KVSDict(os.path.join(self.savepath, 'vecs.ldb'))
        

    def train(self, tagged_texts):
        with self.vecs.write_batch() as wb:
            for tag, text in tagged_texts:
                if not isinstance(tag, str):
                    print('WARNING: text tag is not str (type {})'.format(type(tag)))
                    tag = str(tag)
                self.vecs[tag] = self._calc_vec(text)

    def save(self):
        self.wvmodel.save(os.path.join(self.savepath, 'wvmodel.model'))
        tmp_wvmodel = self.wvmodel
        self.wvmodel = None
        del self.vecs
        with open(os.path.join(self.savepath, 'model_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.init_vecs()
        self.wvmodel = tmp_wvmodel

    def _calc_vec(self, text):
        arr = np.array(
                [self.wvmodel.wv[v] for v in text if v in self.wvmodel.wv]
                )
        if arr.shape == np.array([]).shape:
            return np.zeros(self.wvmodel.wv.vector_size)
        return np.average(arr, axis=0)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'model_class.cls'), "rb") as f:
            self = pickle.load(f)
        self.init_vecs()
        return self

    def load_wvmodel(self, model_path):
        self.wvmodel = Word2Vec.load(model_path)

class WVAverageLayer(ModelLayerBase):
    def train(self, dataset, tokenizer='mecab'):
        if tokenizer == 'spm':
            _tokenizer = get_spm(dataset, self.level, os.path.join(self.savepath, 'spm'))
        else:
            _tokenizer = 'mecab'
        dataset.set_iterator_mode(self.level, tag=False, sentence=True, tokenizer=_tokenizer)
        wv = Word2Vec(
                sentences = dataset,
                size = 500,
                window = 4,
                min_count=10,
                workers=multiprocessing.cpu_count()
                )
        dataset.set_iterator_mode(self.level, tag=True, sentence=True, tokenizer=_tokenizer)
        self.model = WVAverageModel(wv, os.path.join(self.savepath, 'model_body'))
        self.model.train(dataset)

    def save(self):
        self.model.save()
        model = self.model
        self.model = None
        with open(os.path.join(self.savepath, 'layer_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.model = model

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'layer_class.cls'), "rb") as f:
            layer = pickle.load(f)
        layer.model = WVAverageModel.load(os.path.join(path, 'model_body'))
        return layer

    def compare(self, elem1, elem2):
        v1, v2 = self.model.vecs[elem1], self.model.vecs[elem2]
        return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

    def __str__(self):
        return "WVAModel"

class Doc2VecLayer(ModelLayerBase):
    def train(self, dataset):
        if tokenizer == 'spm':
            _tokenizer = get_spm(dataset, self.level, os.path.join(self.savepath, 'spm'))
        else:
            _tokenizer = 'mecab'
        dataset.set_iterator_mode(self.level, gensim=True, tokenizer=_tokenizer)
        self.model = Doc2Vec(
            documents = dataset,
            dm = 1,
            vector_size=500,
            window=4,
            min_count=10,
            workers=multiprocessing.cpu_count()
            )

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'layer_class.cls'), "rb") as f:
            layer = pickle.load(f)
        layer.model = Doc2Vec.load(os.path.join(path, 'model_body'))
        return layer

    def save(self):
        os.makedirs(self.savepath)
        self.model.save(os.path.join(self.savepath, 'model_body'))
        model = self.model
        self.model = None
        with open(os.path.join(self.savepath, 'layer_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.model = model

    def compare(self, elem1, elem2):
        return self.model.docvecs.similarity(elem1, elem2)

    def __str__(self):
        return "Doc2VecModel"

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

class GDBWVAverageModel(WVAverageModel):
    def __init__(self, wvmodel, savepath, gdb, level):
        self.wvmodel = wvmodel
        self.savepath = savepath
        os.makedirs(self.savepath, exist_ok=True)
        self.uri = gdb.uri
        self.auth = gdb.auth
        self.driver = GraphDatabase.driver(uri=self.uri, auth=self.auth)
        self.level = level if isinstance(level, str) else level.__name__


    def train(self, tagged_texts):
        with self.driver.session() as session:
            for tag, text in tagged_texts:
                if not isinstance(tag, str):
                    print('WARNING: text tag is not str (type {})'.format(type(tag)))
                    tag = str(tag)
                session.run(
                        """
                            MATCH (n:%s{id: '%s'})
                            SET n.wva_vec = $vec
                        """ % (self.level+'s', tag), vec=self._calc_vec(text).tolist())

    def __getitem__(self, key):
        pc, mc, fc, num = re.split('/', key)
        with self.driver.session() as session:
            return session.run("""
                    MATCH (p:Prefectures)-->(m:Municipalities)-->(s:Statutories)-[:HAS_ELEM*]->(n:%s)
                    WHERE p.code = $pc
                    AND m.code = $mc
                    AND s.code = $fc
                    AND n.eid = $eid
                    RETURN n.wva_vec
                    LIMIT 1
                    """ % self.level+'s',
                    pc=pc, mc=mc, fc=fc, eid=key).single()[0]

    def save(self):
        self.wvmodel.save(os.path.join(self.savepath, 'wvmodel.model'))
        tmp_wvmodel = self.wvmodel
        self.wvmodel = None
        del self.driver
        with open(os.path.join(self.savepath, 'model_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.wvmodel = tmp_wvmodel

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'model_class.cls'), "rb") as f:
            self = pickle.load(f)
        self.driver = GraphDatabase.driver(uri=self.uri, auth=self.auth)
        return self

    def load_wvmodel(self, model_path):
        self.wvmodel = Word2Vec.load(model_path)

class GDBWVAverageLayer(WVAverageLayer):
    def train(self, dataset):
        dataset.set_iterator_mode(self.level, tag=False, sentence=True)
        wv = Word2Vec(
                sentences = dataset,
                size = 500,
                window = 4,
                min_count=10,
                workers=multiprocessing.cpu_count()
                )
        dataset.set_iterator_mode(self.level, tag=True, sentence=True)
        self.model = GDBWVAverageModel(wv, os.path.join(self.savepath, 'model_body'), dataset.gdb, self.level)
        self.model.train(dataset)

    def compare(self, elem1, elem2):
        return np.dot(self.model[elem1], self.model[elem2])

    def __str__(self):
        return "GDBWVAModel"
