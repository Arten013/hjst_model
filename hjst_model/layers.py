from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import re
import os
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from copy import copy
import shutil
import numpy as np
import pickle
import editdistance
import multiprocessing
from neo4j.v1 import GraphDatabase
from jstatutree import kvsdict
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
try:
    import sentencepiece as spm
except:
    print('WARNING: you cannot sentencepiece model')



class LayerBase(object):
    def refine_scored_pairs(self, scored_pairs, threshold):
        for k1, k2, _ in scored_pairs:
            score = self.compare(k1, k2)
            scored_pairs.add_scored_pair(k1, k2, score)

    def compare(self, elem1, elem2):
        pass

class MethodLayerBase(LayerBase):
    def __init__(self, dataset, level):
        self.dataset = dataset
        self.level = level

    @classmethod
    def is_model(self):
        return False
    
    def compare_by_idvectors(self, vec1, vec2):
        return self.idvector_to_wvmatrix(vec1) * self.idvector_to_wvmatrix(vec2).T
    
    def idvector_to_wvmatrix(self, id_vec):
        return np.matrix([self[_id] for _id in id_vec])

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

    def __getitem__(self, key):
        return self.model[key]

    @classmethod
    def is_model(self):
        return True

    def __str__(self):
        return self.__class__.__name__
    
class AggregationLayerBase(ModelLayerBase):
    def __init__(self, level, savepath):
        super().__init__(level, savepath)
        os.makedirs(self.savepath, exist_ok=True)
        self.init_vecs()

    def init_vecs(self):
        self.vecs = kvsdict.KVSDict(os.path.join(self.savepath, 'vecs.ldb'))
    
    def train(self, dataset, base_layer):
        with self.vecs.write_batch() as wb:
            for pnode, cnodes in dataset.kvsdicts["edges"][self.level].items():
                #print(pnode)
                wb[pnode] = self._calc_vec(np.array([base_layer[n] for n in cnodes]))
                
    def __getitem__(self, key):
        return self.vecs[key]
                
    def _calc_vec(self, matrix):
        raise "You must implement aggregation function"

    def save(self):
        del self.vecs
        with open(os.path.join(self.savepath, 'model_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.init_vecs()

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'model_class.cls'), "rb") as f:
            self = pickle.load(f)
        self.init_vecs()
        return self

    def compare(self, elem1, elem2):
        v1, v2 = self.vecs[elem1], self.vecs[elem2]
        return np.dot(v1, v2)

class AverageAGLayer(AggregationLayerBase):
    def _calc_vec(self, matrix):
        v = np.sum(matrix, axis=0)
        return v/np.linalg.norm(v)

class RandomAGLayer(AggregationLayerBase):
    def _calc_vec(self, matrix):
        print(matrix.shape)
        v = np.random.normal(loc=0, scale=1000, size=(matrix.shape[1],))
        return v/np.linalg.norm(v)
    
def get_spm(dataset, savepath, vocab_size=16000):
    model_path = os.path.join(savepath, 'model.model')
    vocab_path = os.path.join(savepath, 'model.vocab')
    corpus_path = os.path.join(savepath, 'corpus.txt')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        dataset.set_iterator_mode("Sentence", tag=False, sentence=True, tokenizer=lambda x: x)
        with open(corpus_path, 'w') as f:
            f.write('\n'.join(dataset))
        spm.SentencePieceTrainer.Train('--input={} --model_prefix=model --vocab_size={}'.format(corpus_path, vocab_size))
        shutil.move('./model.model', model_path)
        shutil.move('./model.vocab', vocab_path)
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
            print("WARNING: a zero vector allocated for the text below:")
            print(text)
            return np.zeros(self.wvmodel.wv.vector_size)
        v = np.sum(arr, axis=0)
        return v/np.linalg.norm(v)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'model_class.cls'), "rb") as f:
            self = pickle.load(f)
        self.init_vecs()
        return self

    def load_wvmodel(self, model_path):
        self.wvmodel = Word2Vec.load(model_path)

class SWEMLayerBase(ModelLayerBase):
    def __init__(self, level, savepath):
        self.level = level
        self.savepath = savepath
        self.wvmodel_path = os.path.join(self.savepath,'..', 'wvmodel.model')
        self.spm_path = os.path.join(self.savepath,'..', 'spm')
        self.model = None
        self.init_vecs()
    
    def init_vecs(self):
        self.vecs = kvsdict.KVSDict(os.path.join(self.savepath, 'vecs.ldb'))
    
    def train(self, dataset, tokenizer='mecab', vocab_size=16000):
        if tokenizer == 'spm':
            spm = get_spm(dataset, self.spm_path, vocab_size=int(vocab_size))
            _tokenizer = lambda x: [w for w in spm.EncodeAsPieces(x)]
        else:
            _tokenizer = 'mecab'
        if os.path.exists(self.wvmodel_path):
            self.load_wvmodel()
        else:
            dataset.set_iterator_mode("Law", tag=False, sentence=True, tokenizer=_tokenizer)
            self.wvmodel = Word2Vec(
                    sentences = dataset,
                    size = 500,
                    window = 4,
                    min_count=10,
                    max_vocab_size=vocab_size,
                    workers=multiprocessing.cpu_count()
                    )
            self.wvmodel.save(self.wvmodel_path)
        dataset.set_iterator_mode(self.level, tag=True, sentence=True, tokenizer=_tokenizer)
        with self.vecs.write_batch() as wb:
            for tag, text in dataset:
                if not isinstance(tag, str):
                    print('WARNING: text tag is not str (type {})'.format(type(tag)))
                    tag = str(tag)
                wb[tag] = self._calc_vec(text)
    
    def _calc_vec(self, text):
        raise "Not implemented"

    def save(self):
        tmp_wvmodel = self.wvmodel
        self.wvmodel = None
        del self.vecs
        with open(os.path.join(self.savepath, 'layer_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.init_vecs()
        self.wvmodel = tmp_wvmodel

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'layer_class.cls'), "rb") as f:
            layer = pickle.load(f)
        layer.init_vecs()
        return layer

    def compare(self, elem1, elem2):
        return np.dot(self.vecs[elem1], self.vecs[elem2])

    def __getitem__(self, key):
        return self.vecs[key]
    
    def load_wvmodel(self):
        self.wvmodel = Word2Vec.load(self.wvmodel_path)
    
    def __str__(self):
        return "WVAModel"

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

class WVAverageLayer(ModelLayerBase):
    def train(self, dataset, tokenizer='mecab', vocab_size=16000):
        if tokenizer == 'spm':
            spm = get_spm(dataset, os.path.join(self.savepath, 'spm'), vocab_size=int(vocab_size))
            _tokenizer = lambda x: [w for w in spm.EncodeAsPieces(x)]
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
        return np.dot(v1, v2)

    def __getitem__(self, key):
        return self.model.vecs[key]
    
    def __str__(self):
        return "WVAModel"

class RandomLayer(ModelLayerBase):
    def __init__(self, level, savepath):
        super().__init__(level, savepath)
        os.makedirs(self.savepath, exist_ok=True)
        self.init_vecs()

    def init_vecs(self):
        self.vecs = kvsdict.KVSDict(os.path.join(self.savepath, 'vecs.ldb'))
    
    def train(self, dataset, scale=1, size=500):
        dataset.set_iterator_mode(self.level, tag=True, sentence=False)
        with self.vecs.write_batch() as wb:
            for tag in list(dataset):
                #print(tag)
                v = np.random.normal(loc=0, scale=float(scale), size=(int(size),))
                wb[tag] = v/np.linalg.norm(v)
                
    def __getitem__(self, key):
        return self.vecs[key]

    def save(self):
        del self.vecs
        with open(os.path.join(self.savepath, 'model_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.init_vecs()

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'model_class.cls'), "rb") as f:
            self = pickle.load(f)
        self.init_vecs()
        return self

    def compare(self, elem1, elem2):
        v1, v2 = self.vecs[elem1], self.vecs[elem2]
        return np.dot(v1, v2)
    
class Doc2VecLayer(ModelLayerBase):
    def train(self, dataset, tokenizer='mecab', vocab_size=16000):
        if tokenizer == 'spm':
            spm = get_spm(dataset, os.path.join(self.savepath, 'spm'), vocab_size=int(vocab_size))
            _tokenizer = lambda x: [w for w in spm.EncodeAsPieces(x)]
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
        
    def __getitem__(self, key):
        return self.model.docvecs[key]

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'layer_class.cls'), "rb") as f:
            layer = pickle.load(f)
        layer.model = Doc2Vec.load(os.path.join(path, 'model_body'))
        return layer

    def save(self):
        os.makedirs(self.savepath, exist_ok=True)
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
    def train(self, dataset, tokenizer='mecab', vocab_size=16000, idf=True, lsi_size=None):
        idf = bool(idf)
        lsi_size = int(lsi_size) if lsi_size else None
        vocab_size = int(vocab_size)
        if tokenizer == 'spm':
            spm = get_spm(dataset, os.path.join(self.savepath, 'spm'), vocab_size=int(vocab_size))
            _tokenizer = lambda x: [w for w in spm.EncodeAsPieces(x)]
        else:
            _tokenizer = 'mecab'
        dataset.set_iterator_mode(self.level, tag=True, sentence=False, tokenizer=_tokenizer)
        self.tag_idx_dict = {k:i for i, k in enumerate(dataset)}
        dataset.set_iterator_mode(self.level, tag=False, sentence=True, tokenizer=_tokenizer)
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
        self.mat = self.transformer.fit_transform([" ".join(s) for s in dataset])

    def save(self):
        os.makedirs(self.savepath, exist_ok=True)
        tmp_mat = self.mat
        with open(os.path.join(self.savepath, 'matrix.npy'), "wb") as f:
            np.save(f, self.mat)
            del self.mat
        with open(os.path.join(self.savepath, 'layer_class.cls'), "wb") as f:
            pickle.dump(self, f)
        self.mat = tmp_mat
        
    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'layer_class.cls'), "rb") as f:
            layer = pickle.load(f)
        with open(os.path.join(path, 'matrix.npy'), "rb") as f:
            layer.mat = np.load(f)
        return layer

    def __getitem__(self, key):
        return self.mat[self.tag_idx_dict[key]]

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
