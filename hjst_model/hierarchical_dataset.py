from jstatutree.mltree.ml_lawdata import JStatutreeKVS
from jstatutree.mltree import ml_etypes
from jstatutree.xmltree import xml_lawdata, xml_etypes
from jstatutree.graphtree import graph_etypes, graph_lawdata
from jstatutree.myexceptions import *
from jstatutree.etypes import sort_etypes, Sentence
from jstatutree.kvsdict import KVSDict
from gensim.models.doc2vec import TaggedDocument

import multiprocessing
from time import time
import traceback
import concurrent
import plyvel
import os
import re
from pathlib import Path
import threading
import queue
from .tokenizer import Morph

def find_all_files(directory, extentions=None):
    for root, dirs, files in os.walk(directory):
        if extentions is None or os.path.splitext(root)[1] in extentions:
            yield root
        for file in files:
            if extentions is None or os.path.splitext(file)[1] in extentions:
                yield os.path.join(root, file)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import plyvel
from queue import Queue
import shutil
import re
import pickle

class ReikiKVSDictDB(object):
    def __init__(self, basepath, thread_num=5):
        self.basepath = Path(basepath)
        self.thread_num = thread_num
        self.workers = ThreadPoolExecutor(max_workers=self.thread_num)
    
    def remove_files(self):
        shutil.rmtree(self.basepath)
    
    mcode_ptn = re.compile('\d{6}')
    @property
    def mcodes(self):
        if not self.basepath.exists():
            return []
        return [x.name for x in self.basepath.iterdir() if x.is_dir() and re.match(self.__class__.mcode_ptn, x.name)]
        
    def get_db_path(self, mcode):
        return self.basepath / Path(mcode)
        
    def get_db(self, mcode):
        dbpath = self.get_db_path(mcode)
        if not dbpath.exists():
            os.makedirs(str(dbpath))
        return  plyvel.DB(str(dbpath), create_if_missing=True)
        
    def __getitem__(self, key):
        ret = self.get(key)
        if ret is None:
            raise KeyError(key, 'is not registered in the db', self.basepath)
        return ret

    def get(self, key, default=None):
        mcode, item_key = self._encode_key(key)
        if not self.get_db_path(mcode).exists():
            return default
        db = self.get_db(mcode)
        ret =  pickle.loads(db.get(item_key, default=default))
        db.close()
        return ret
    
    def put(self, key, value):
        mcode, item_key = self._encode_key(key)
        db = self.get_db(mcode)
        db.put(item_key, pickle.dumps(value))
        db.close()
    
    def delete(self, key):
        mcode, item_key = self._encode_key(key)
        db = self.get_db(mcode)
        db.delete(item_key)
        for i in db.iterator(include_key=True, include_value=False):
            break
        else:
            shutli.rmtree(self.get_db_path(mcode))
        db.close()
    
    @staticmethod
    def _encode_key(key):
        parts = Path(key).parts
        return parts[1], os.path.join(*parts[2:]).encode()
    
    @staticmethod
    def _decode_key(mcode, item_key):
        return os.path.join(mcode[:2], mcode, item_key.decode())
    
    def is_empty(self):
        return len(self.mcodes) == 0
    
    def items(self):
        for mcode in sorted(self.mcodes, key=lambda x:int(x)):
            db = self.get_db(mcode)
            for item_key, value in db.iterator(include_key=True, include_value=True):
                yield self._decode_key(mcode, item_key), pickle.loads(value)

    def keys(self):
        for mcode in sorted(self.mcodes, key=lambda x:int(x)):
            db = self.get_db(mcode)
            for item_key in db.iterator(include_key=True, include_value=False):
                yield self._decode_key(mcode, item_key)

    def values(self):
        for mcode in sorted(self.mcodes, key=lambda x:int(x)):
            db = self.get_db(mcode)
            for value in db.iterator(include_key=False, include_value=True):
                yield pickle.loads(value)
    
    def _qiter_base(self, enqueue_func):
        queue = Queue()
        enqueue_func.queue = queue
        mcodes = self.mcodes
        for mcode in sorted(mcodes, key=lambda x:int(x)):
            self.workers.submit(enqueue_func, mcode)
        running_count = len(mcodes)
        while running_count > 0:
            item = queue.get()
            if item is None:
                running_count -= 1
            else:
                yield item
    
    def qitems(self):
        def enqueue_func(mcode):
                queue = enqueue_func.queue
                db = self.get_db(mcode)
                for item_key, value in db.iterator(include_key=True, include_value=True):
                    queue.put((self._decode_key(mcode, item_key), pickle.loads(value)))
                queue.put(None)
        for k, v in self._qiter_base(enqueue_func):
            yield k, v

    def qkeys(self):
        def enqueue_func(mcode):
                queue = enqueue_func.queue
                db = self.get_db(mcode)
                for item_key in db.iterator(include_key=True, include_value=False):
                    queue.put(self._decode_key(mcode, item_key))
                queue.put(None)
        yield from self._qiter_base(enqueue_func)

    def qvalues(self):
        def enqueue_func(mcode):
                queue = enqueue_func.queue
                db = self.get_db(mcode)
                for value in db.iterator(include_key=False, include_value=True):
                    queue.put(pickle.loads(value))
                queue.put(None)
        yield from self._qiter_base(enqueue_func)

class BatchReikiWriter(object):
    def __init__(self, reikikvsdb, *wb_args, **wb_kwargs):
        self.db = reikikvsdb
        self.wb_args = wb_args
        self.wb_kwargs = wb_kwargs
        self.wbdict = dict()
        self.dbdict = dict()
    
    def __setitem__(self, key, value):
        mcode, item_key = self.db._encode_key(key)
        if mcode not in self.wbdict:
            db = self.db.get_db(mcode)
            self.wbdict[mcode] = db.write_batch(*self.wb_args, **self.wb_kwargs)
            self.dbdict[mcode] = db
        #print(mcode, item_key, value)
        self.wbdict[mcode].put(item_key, pickle.dumps(value))

    def __delitem__(self, key):
        mcode, item_key = self.db._encode_key(key)
        if mcode not in self.wbdict():
            db = self.db.get_db(mcode)
            self.wbdict[mcode] = db.write_batch(*self.wb_args, **self.wb_kwargs)
            self.dbdict[mcode] = db
        self.wbdict[mcode].delete(item_key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self.write()
        return True
    
    
    def write(self):
        for wb in self.wbdict.values():
            wb.write()
        self.wbdict = {}
        for mcode, db in self.dbdict.items():
            for i in db.iterator(include_key=True, include_value=False):
                db.close()
                break
            else:
                db.close()
                shutli.rmtree(self.db.get_db_path(mcode))
        self.dbdict = {}
    

class ReikiKVSDict(object):
    ENCODING = "utf8"

    def __init__(self, path, thread_num=None, create_if_missing=True):
        self.path = path
        self.thread_num = thread_num or 5
        self.db = ReikiKVSDictDB(self.path, self.thread_num)

        if create_if_missing:
            os.makedirs(self.path, exist_ok=True)
    
    @property
    def path(self):
        if "_path" not in self.__dict__:
            self._path = None
        return self._path

    @path.setter
    def path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.splitext(path)[1] == "":
            path += ".ldb"
        self._path = path
        
    def to_dict(self):
        return {k:v for k, v in self.items()}

    def __setitem__(self, key, val):
        self.db.put(key, val)

    def __getitem__(self, key):
        return self.db.get(key)

    def __delitem__(self, key):
        self.db.delete(key)

    def get(self, key, default=None):
        return self.db.get(key, default=default)
        
    def __len__(self):
        l = 0
        for _ in self.qkeys():
            l += 1
        return l
    
    def is_prefixed_db(self):
        return False

    def is_empty(self):
        return self.db.is_empty()
    
    def write_batch_mapping(self, mapping, *args, **kwargs):
        with BatchReikiWriter(self.db) as wv:
            for k, v in mapping.items():
                wv[k] = v
            
    def write_batch(self, *args, **kwargs): 
        return BatchReikiWriter(self.db, *args, **kwargs)
    
    def items(self):
        return self.db.items()

    def keys(self):
        return self.db.keys()

    def values(self):
        return self.db.values()

    def qitems(self):
        return self.db.qitems()

    def qkeys(self):
        return self.db.qkeys()

    def qvalues(self):
        return self.db.qvalues()




import re, pickle
class Lawcodes(object):
    def __init__(self, path):
        self.basepath = Path(path)
        self.lcdict = {}
        self.changed_list = []
    
    mcode_ptn = re.compile('\d{6}')
    @property
    def mcodes(self):
        yield from self.changed_list
        if not self.basepath.exists():
            return []
        yield from (x.stem for x in self.basepath.iterdir() if not x.is_dir() and re.match(self.__class__.mcode_ptn, x.stem) and x.stem not in self.changed_list)

    def get_path(self, mcode=None, pickle_file=False):
        if mcode:
            mcode = str(mcode)
            if pickle_file:
                return self.basepath/Path(mcode+'.pkl')
            else:
                return self.basepath/Path(mcode)
        else:
            return self.basepath
    
    def __contains__(self, item):
        parts = Path(item).parts
        if len(parts) < 3:
            return False
        pcode, mcode, fcode = parts[:3]
        self.lcdict[pcode] = self.lcdict.get(pcode, dict())
        self.lcdict[pcode][mcode] = self.lcdict[pcode].get(mcode, self._load_sub(mcode)) 
        return fcode in self.lcdict[pcode][mcode] 
    
    def _load_sub(self, mcode):      
        path = self.get_path(mcode, True)
        if not path.exists() or not os.path.getsize(path):
            return []
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __iter__(self):
        for mcode in self.mcodes:
            pcode = mcode[:2]
            self.lcdict[pcode] = self.lcdict.get(pcode, dict())
            d = self.lcdict[pcode].get(mcode, self._load_sub(mcode))
            yield from (os.path.join(pcode, mcode, fcode) for fcode in d)

    def __len__(self):
        return sum(1 for v in self)

    def append(self, key):
        if key in self:
            return
        parts = Path(key).parts
        if len(parts) != 3:
            return
        pcode, mcode, fcode = parts
        self.lcdict[pcode] = self.lcdict.get(pcode, dict())
        self.lcdict[pcode][mcode] =self.lcdict[pcode].get(mcode, self._load_sub(mcode)) + [fcode]
        if mcode not in self.changed_list:
            self.changed_list.append(mcode)
    
    def __getitem__(self, key):
        key = str(key)
        if re.match('\d{2}$', key):
            return self.lcdict(key)
        if re.match('\d{6}$', key):
            try:
                return self.lcdict[key[:2]][key]
            except KeyError:
                raise KeyError(key)
        if re.match('(\d{2}/)\d{6}$', key):
            mcode, fcode = Path(key).parts
            try:
                return self.lcdict[mcode][fcode]
            except KeyError:
                raise KeyError(key)
        raise KeyError(key)
    
    def write(self):
        os.makedirs(self.basepath, exist_ok=True)
        for mcode in self.changed_list:
            with open(self.get_path(mcode, True), 'wb') as f:
                pickle.dump(self[mcode], f)
                
class HierarchicalDataset(object):
    def __init__(self, dbpath, dataset_name, levels, only_reiki=True, only_sentence=True, *args, **kwargs):
        self.path = os.path.abspath(dbpath)
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.levels = sort_etypes(levels)
        self.dataset_name = dataset_name
        self._closed = False

        self.kvsdicts = dict()
        self.lawcodes = Lawcodes(os.path.join(self.path, self.dataset_name, 'lawcodes'))
        self.kvsdicts["texts"] = {
            l.__name__: ReikiKVSDict(path=os.path.join(self.path, self.dataset_name, "texts", l.__name__)) for l in self.levels
            }
        self.kvsdicts["edges"] = {
            l.__name__: ReikiKVSDict(path=os.path.join(self.path, self.dataset_name, "edges", l.__name__)) for l in self.levels[:-1]
            }
        self.kvsdicts['edges']['Statutory'] = ReikiKVSDict(path=os.path.join(dbpath, dataset_name, "edges", 'Statuotory'))

    def is_closed(self):
        return self._closed

    def __getitem__(self, key):
        return self.kvsdicts[key]
  
    def get_tags(self, roots, level):
        for root in roots:
            current_level = re.split('\(', os.path.split(root)[1])[0]
            if level == current_level:
                yield root
            else:
                yield from self.get_tags(self.kvsdicts['edges'][current_level][root], level) 

       
    def close(self):
        self._closed = True
        pass

    def iter_lawcodes(self):
        yield from self.lawcodes

    def __len__(self):
        return len(self.lawcodes)

    def set_data(self, reader):
        if self.only_reiki and not reader.lawdata.is_reiki():
            return False
        code = reader.lawdata.code
        statutree = reader.get_tree()
        writers = {
                l:self.kvsdicts["texts"][l.__name__].write_batch(transaction=True)
                for l in self.levels
            }
        for li, level in enumerate(self.levels):
            next_elems = list(statutree.depth_first_search(level, valid_vnode=True))
            if li == 0:
                self.kvsdicts['edges']['Statutory'][reader.lawdata.code] = [e.code for e in next_elems]
            for elem in next_elems:
                #print("reg:", "vnode" if elem.is_vnode else "node", elem.code)
                if self.only_sentence:
                    writers[level][elem.code] = "".join(elem.iter_sentences()) 
                else:
                    writers[level][elem.code] = "".join(elem.iter_texts())
                if level != self.levels[-1]:
                    self.kvsdicts["edges"][level.__name__][elem.code] = [e.code for e in elem.depth_first_search(self.levels[li+1], valid_vnode=True)]
        for writer in writers.values():
            writer.write()
        return True
    
    def register_directory(self, basepath, **kwargs):
        codes = [os.path.join(*Path(p).with_suffix('').parts[-3:]) for p in Path(basepath).glob('**/*.xml')]
        basedir = os.path.join(*next(Path(basepath).glob('**/*.xml')).parts[:-3])
        self.register_from_codes(codes=codes, basedir=basedir, **kwargs)
    
    @staticmethod
    def keywords_search(keywords, text):
        for kw in keywords:
            if kw not in text:
                return False
        return True
    
    @staticmethod
    def _get_rr(codes, basedir, exist_lawcodes, queue, close_signal, keywords=None):
        keywords = keywords or []
        for code in codes:
            if close_signal.is_set():
                break
            if code in exist_lawcodes:
                continue
            path = os.path.join(basedir, code+'.xml')
            rr = xml_lawdata.ReikiXMLReader(path)
            rr.open()
            if rr.is_closed():
                continue
            if not HierarchicalDataset.keywords_search(keywords, rr.lawdata.name):
                continue 
            
            try:
                list(rr.get_tree().depth_first_iteration([Sentence]))
                #print('add:', rr.lawdata.name)
                queue.put(rr)
            except LawError as e:
                #print(e)
                
                #print('add:', rr.lawdata.name)
                continue
        else:
            queue.put(None)
            close_signal.set()
            
    def register_from_codes(self, codes, basedir, maxsize=None, keywords=None, **kwargs):
        # get lawcodes in the dataset
        rr_queue = queue.Queue()
        close_signal  = threading.Event()
        thread = threading.Thread(target=self._get_rr, args=(codes, basedir, self.lawcodes, rr_queue, close_signal, keywords))
        thread.start()
        writers = {}
        writer_list = []
        for k in self.kvsdicts.keys():
            #print(k, self.kvsdicts[k])
            writers[k] = {l:v.write_batch() for l, v in self.kvsdicts[k].items()}
            writer_list.extend(writers[k].values())
        while True:
            item = rr_queue.get()
            if item is None:
                #print('BREAK')
                break
            rr = item
            if maxsize is not None and len(self.lawcodes) >= maxsize:
                print(self.lawcodes)
                print(len(self.lawcodes))
                close_signal.set()
                thread.join()
                break
            reader = rr
            #print('Check:', rr.lawdata.name, rr.lawdata.code)
            if self.only_reiki and not reader.lawdata.is_reiki():
                #print('Not Reiki')
                continue
            #print('Reiki')
            code = reader.lawdata.code
            #root = ETYPES[0](self.lawdata)
            #root.root = self.root_etree
            #return root
            #print(rr.root_etree)
            statutree = reader.get_tree()
            for li, level in enumerate(self.levels):
                next_elems = list(statutree.depth_first_search(level, valid_vnode=True))
                if li == 0:
                    writers['edges']['Statutory'][reader.lawdata.code] = [e.code for e in next_elems]
                for elem in next_elems:
                    #print("reg:", "vnode" if elem.is_vnode else "node", elem.code)
                    levelstr = level if isinstance(level, str) else level.__name__
                    if self.only_sentence:
                        writers['texts'][levelstr][elem.code] = "".join(elem.iter_sentences()) 
                    else:
                        writers['texts'][levelstr][elem.code] = "".join(elem.iter_texts())
                    if level != self.levels[-1]:
                        writers["edges"][levelstr][elem.code] = [e.code for e in elem.depth_first_search(self.levels[li+1], valid_vnode=True)]
            self.lawcodes.append(rr.lawdata.code)
            print('reg:', item.get_tree())
            rr.close()
        self.lawcodes.write()
        self.is_empty = False
        for writer in writer_list:
              writer.write()

    def set_iterator_mode(self, level, tag=None, sentence=None, tokenizer='mecab', gensim=False, multithread=False):
        self.itermode_level = level
        self.itermode_tag = tag if tag is not None else self.__dict__.get('itermode_tag', True)
        self.itermode_sentence = sentence if sentence is not None else self.__dict__.get('itermode_sentence', True)
        self.tokenizer = Morph().surfaces if tokenizer == 'mecab' else tokenizer
        self.multithread = multithread
        self.gensim = gensim

    def __iter__(self):
        assert self.__dict__.get('itermode_level', False), 'You must call set_iterator_mode before call iterator'
        d = self.kvsdicts['texts'][self.itermode_level]
        if self.gensim:
            self.generator = map(lambda x: TaggedDocument(self.preprocess(x[1]), [x[0]]), getattr(d, 'qitems' if self.multithread else 'items')())
        elif self.itermode_tag and self.itermode_sentence:
            self.generator = map(lambda x: (x[0], self.preprocess(x[1])), getattr(d, 'qitems' if self.multithread else 'items')())
        elif self.itermode_sentence:
            self.generator = map(lambda x: self.preprocess(x), getattr(d, 'qvalues' if self.multithread else 'values')())
        elif self.itermode_tag:
            self.generator = getattr(d, 'qkeys' if self.multithread else 'keys')()
        else:
            self.generator = (x for x in [])
        return self

    def __next__(self):
        return next(self.generator)
    def iter_tagged_sentence(self, level):
        yield from ((t, self.preprocess(s)) for t, s in self.kvsdicts["texts"][level].items())

    def iter_gensim_tagged_documents(self, level, _tokenizer='mecab'):
        return DatasetGensimGenerator(kvsdict=self.kvsdicts['texts'][level], preprocess=self.preprocess)

    def preprocess(self, sentence):
        return self.tokenizer(sentence)

class HierarchicalGraphDataset(object):
    def __init__(self, loginkey, dataset_name, levels='ALL', only_reiki=True, only_sentence=False, *args, **kwargs):
        self.only_reiki = only_reiki
        self.only_sentence = only_sentence
        self.gdb = graph_lawdata.ReikiGDB(**loginkey)
        self.morph_separator = Morph()
        self.levels = levels
        self.name= dataset_name
        with self.gdb.driver.session() as session:
            self.root_codes = [v[0] for v in session.run("MATCH (n:%s) RETURN n.code" % self.name).values()]

    @classmethod
    def init_by_config(cls, config):
        return cls(
                    loginkey=config.loginkey,
                    dataset_name=config.section.name,
                    levels=config['levels'],
                    only_reiki=config['only_reiki'],
                    only_sentence=config['only_reiki']
                )

    def add_government(self, govcode):
        with self.gdb.driver.session() as session:
            if len(govcode) == 2:
                res = session.run("""
                        MATCH (pref:Prefectures{code: '%s'})-[:PREF_OF]->(muni:Municipalities)
                        SET muni:%s
                        RETURN muni.code
                        """ % (govcode, self.name)
                        ).values()
                self.root_codes.extend([i[0] for i in res])
            else:
                session.run("""
                        MATCH (muni:Municipalities{code: '%s'})
                        SET muni:%s
                        """ % (govcode, self.name)
                        )
                self.root_codes.append(govcode)
            self.root_codes = list(set(self.root_codes))

    def set_iterator_mode(self, level, tag=None, sentence=None, gensim=None):
        self.itermode_level = level
        self.itermode_tag = tag if tag is not None else self.__dict__.get('itermode_tag', None)
        self.itermode_sentence = sentence if sentence is not None else self.__dict__.get('itermode_sentence', None)
        self.itermode_gensim = gensim if gensim is not None else self.__dict__.get('itermode_gensim', None)

    def __iter__(self):
        assert self.__dict__.get('itermode_level', False), 'You must call set_iterator_mode before call iterator'
        def get_element_nodes(code):
            level_label = (self.itermode_level.capitalize() if isinstance(self.itermode_level, str) else self.itermode_level.__name__) + 's'
            with self.gdb.driver.session() as session:
                ret = [{'id': i[0], 'fulltext': i[1]} for i in session.run("""MATCH (muni:Municipalities{code: '%s'})-[]->(:Statutories)-[*1..]->(elem:%s) RETURN elem.id, elem.fulltext""" % (code, level_label))]
            return ret
        node_gen = (enode for enode_gen in (get_element_nodes(code) for code in self.root_codes) for enode in enode_gen)
        
        if self.itermode_tag and self.itermode_sentence:
            if self.itermode_gensim:
                self.generator = map(lambda x: TaggedDocument(tags=[x['id']], words=self.preprocess(x['fulltext'])), node_gen)
            else:
                self.generator = map(lambda x: (x['id'], self.preprocess(x['fulltext'])), node_gen)
        elif self.itermode_sentence:
            self.generator = map(lambda x: self.preprocess(x['fulltext']), node_gen)
        elif self.itermode_tag:
            self.generator = map(lambda x: x['id'], node_gen)
        else:
            self.generator = (x for x in [])
        return self

    def __next__(self):
        return next(self.generator)

    def iter_tagged_sentence(self, level):
        self.set_iterator_mode(level, tag=True, sentence=True)
        return self

    def iter_gensim_tagged_documents(self, level):
        self.set_iterator_mode(level, tag=True, sentence=True, gensim=True)
        return self

    def preprocess(self, sentence):
        return self.morph_separator.surfaces(sentence)

class DatasetGensimGenerator(object):
    def __init__(self, kvsdict, preprocess):
        self.kvsdict = kvsdict
        self.preprocess = preprocess

    def __iter__(self):
        self.gen = (TaggedDocument(self.preprocess(s), [t]) for t, s in self.kvsdict.items())
        return self

    def __next__(self):
        return  next(self.gen)

