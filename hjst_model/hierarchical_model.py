import itertools
import numpy as np
from time import time
import gc
from .scored_pair import ScoredPairGDB
import concurrent
import pandas as pd
from queue import Queue
import nmslib
import re
import os
from pathlib import Path



class HierarchicalModel(object):
    def __init__(self):
        self.layers = dict()
        self.thresholds = dict()
        self.levels = []

    def create_index(self, level, tags):
        vectors = np.matrix([self.layers[level][tag] for tag in tags])
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(vectors)
        index.createIndex({'post': 2}, print_progress=True)
    
        return vectors, index

    def comptable(self, dataset, query, targets, threshold, level):
        qlevel = re.split('\(', os.path.split(query)[1])[0]
        
        # construct target sentence space
        target_spaces = dict()
        target_tags = dict()
        for ltag in targets:
            #dataset.set_iterator_mode(level=level, tag=True, sentence=False)
            target_tags[ltag] = list(dataset.get_tags([ltag], level))
            target_vectors, target_spaces[ltag] = self.create_index(level, target_tags[ltag])

        columns = []
        name = str(dataset.get_elem(query))
        columns.append(("{0}({1})".format(name, query), level))
        for ltag in targets:
            tag = ltag#os.path.split(ltag)[0]
            name = str(dataset.get_elem(tag))
            columns.append(("{0}({1})".format(name, tag), level))
            columns.append(("{0}({1})".format(name, tag), 'Distance'))
        df = pd.DataFrame(columns=columns)
        df.columns = pd.MultiIndex.from_tuples(columns)

        sid = 0
        for j, s in enumerate(dataset.get_tags([query], level)):
            sid += 1
            df.loc[sid, columns[0]] = dataset.kvsdicts["texts"][level][s]
            for l, ltag in enumerate(targets):
                resi, resd  = target_spaces[ltag].knnQuery(self.layers[level][s], 1)
                if resd[0] < threshold:
                    df.loc[sid, (columns[2*l+1][0], level)] = dataset.kvsdicts["texts"][level][target_tags[ltag][resi[0]]]
                    df.loc[sid, (columns[2*l+1][0], 'Distance')] = round(resd[0], 3)
        return df
    
    def topk_comptable(self, dataset, query, k, threshold, level, root_threshold=None):
        qlevel = re.split('\(', os.path.split(query)[1])[0]
        dataset.set_iterator_mode(level=qlevel, tag=True, sentence=False)
        law_tags = list(dataset)
        _, law_index = self.create_index(qlevel, law_tags)
        tidx, similarity = law_index.knnQuery(self.layers[qlevel][query], k=k+1)
        target_law_tags = []
        for i, sim in zip(tidx, similarity):
            if law_tags[i] != query:
                continue
            if root_threshold and sim < root_threshold:
                print("not enough distance", dataset.get_elem(law_tags[i]))
                k -= 1
                continue
            target_law_tags.append(law_tags[i])
        return self.comptable(dataset, query, target_law_tags[:k], threshold, level)
    
    
    def set_trained_model_layer(self, level, model, threshold):
        print("set layer:", model.__class__.__name__, threshold)
        assert level not in self.levels, "Level " + str(level)+ " has already exists"
        self.levels.append(level)
        self.layers[level] = model
        self.thresholds[level] = threshold
        
    def get_expanded_layer(self, testset, tree):
        paths = []
        edge_queue = Queue()
        #print(tree)
        for edge in testset.kvsdicts['edges']['Statutory'][tree]:
            edge_queue.put([edge])
        while not edge_queue.empty():
            item = edge_queue.get()
            depth = len(item)
            if depth < len(self.levels):
                edges = testset.kvsdicts['edges'][self.levels[depth-1]][item[-1]]
                for edge in edges:
                    edge_queue.put(item+[edge])
            else:
                paths.append(item)
        return np.array(paths).transpose()
        
    def get_layered_law_matrix(self, testset, r):
        print(self.levels, self.layers)
        layered_idvec = self.get_expanded_layer(testset, r)
        try:
            layered_law_matrix = np.array([self.layers[self.levels[i]].idvector_to_wvmatrix(m) for i, m in enumerate(layered_idvec)])
        except:
            layered_law_matrix = np.array(
                [
                    np.matrix([self.layers[self.levels[i]][_id] for _id in id_vec])
                    for i, id_vec in enumerate(layered_idvec)
                ])
            
        return layered_idvec, layered_law_matrix

    def refine_pairs(self, testset, candidate_pairs):
        for li, level in enumerate(self.levels):
            candidate_num = len(candidate_pairs[level])
            t1 = time()
            self.layers[level].refine_scored_pairs(candidate_pairs[level], threshold=self.thresholds[level])
            t2=time()
            print("refining_time({}):".format(level), t2-t1, "sec")
            if level == self.levels[-1]:
                print("threshold pass: {0:,}/{1:,}".format(len([score for _, _, score in candidate_pairs[level] if score > self.thresholds[level]]), candidate_num))
                return candidate_pairs[level]
            si = 0
            for k1, k2, score in candidate_pairs[level]:
                if self.thresholds[level] is not None and score <= self.thresholds[level]:
                    continue
                candidate_pairs[self.levels[li+1]].add_by_iterable_pairs(
                    itertools.product(
                            testset["edges"][level][k1],
                            testset["edges"][level][k2]
                        ) 
                ) 
                si += 1
            print("calculating next pairs:", time()-t2, "sec")
            print("threshold pass: {0:,}/{1:,}".format(si, candidate_num))
        raise("Unexpected error")

    def get_pairs(self, testset, dbpath):
        candidate_pairs = LeveledScoredPairKVS(path=dbpath, levels=self.levels)
        candidate_pairs.add_by_iterable(itertools.combinations(testset.sentence_dict[self.levels[0]].keys(), 2))
        return self.refine_pairs(testset, candidate_pairs)

    def gdb_refine_core(self, candidate_pairs, e1, e2, level):
        score = self.layers[level].compare(e1, e2)
        if score > self.thresholds[level]:
            return 1, len(candidate_pairs.add_passed_pair(e1, e2, score))
        else:
            candidate_pairs.add_not_passed_pair(e1, e2, score)
            return 0, 0

    def refine_gdb_pairs(self, loginkey, dataset_name, experiment_name, limit=None):
        candidate_pairs = ScoredPairGDB(loginkey, experiment_name, self.levels[0], dataset_name)
        candidate_num  =candidate_pairs.init_candidate_edge(limit)
        for li, level in enumerate(self.levels):
            print('[ level:', level.__name__, ']')
            print('candidate pair num: {0:,}'.format(candidate_num))
            t1 = time()
            passed_count = 0
            next_candidate_num = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                        executor.submit(self.gdb_refine_core, candidate_pairs, e1, e2, level)
                        for e1, e2 in candidate_pairs.iter_candidate_pairs()
                        ]
                concurrent.futures.wait(futures)
                for future in futures:
                    pc, cn = future.result()
                    passed_count += pc
                    next_candidate_num += cn
            t2=time()
            print("threshold pass: {0:,}({1:}%)".format(passed_count, round(passed_count/candidate_num*100, 2) if candidate_num>0 else 'N/A', 2))
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            candidate_num = next_candidate_num
            if len(self.levels) == li+1:
                break
            candidate_pairs = ScoredPairGDB(loginkey, experiment_name, self.levels[li+1], dataset_name)
        raise("Unexpected error")

    def gdbwva_refine_core(self, candidate_pairs, e1, v1, e2, v2, level):
        score = np.dot(np.array(v1), np.array(v2))
        if score > self.thresholds[level]:
            return 1, len(candidate_pairs.add_passed_pair(e1, e2, score))
        else:
            candidate_pairs.add_not_passed_pair(e1, e2, score)
            return 0, 0

    def refine_gdbwva_pairs(self, loginkey, dataset_name, experiment_name, limit=None):
        candidate_pairs = ScoredPairGDB(loginkey, experiment_name, self.levels[0], dataset_name)
        candidate_num  =candidate_pairs.init_candidate_edge(limit)
        for li, level in enumerate(self.levels):
            print('[ level:', level.__name__, ']')
            print('candidate pair num: {0:,}'.format(candidate_num))
            t1 = time()
            passed_count = 0
            next_candidate_num = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                        executor.submit(self.gdb_refine_core, candidate_pairs, e1, v1, e2, v2, level)
                        for e1, v1, e2, v2 in candidate_pairs.iter_candidate_pairs(vecs=True)
                        ]
                concurrent.futures.wait(futures)
                for future in futures:
                    pc, cn = future.result()
                    passed_count += pc
                    next_candidate_num += cn
            t2=time()
            print("threshold pass: {0:,}({1:}%)".format(passed_count, round(passed_count/candidate_num*100, 2) if candidate_num>0 else 'N/A', 2))
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            candidate_num = next_candidate_num
            if len(self.levels) == li+1:
                break
            candidate_pairs = ScoredPairGDB(loginkey, experiment_name, self.levels[li+1], dataset_name)
        raise("Unexpected error")

class BTHierarchicalModel(object):
    def __init__(self, trainingset):
        self.trainingset = trainingset
        self.levels = trainingset.levels

    def set_layer(self, level, layer_cls, savepath, threshold=0.3):
        assert level in self.levels, "This level is not in levels"+str(level)
        if layer_cls.is_model():
            try:
                self.layers[level] = layer_cls.load(savepath)
                print("Load layer:", savepath)
            except:
                self.layers[level] = layer_cls(level, savepath)
                print("Create layer.")
        else:
            self.layers[level] = layer_cls(self.trainingset, level)
        self.thresholds[level] = threshold

    def batch_training(self):
        assert len(self.layers) == len(self.levels), "You must register layers for all levels."
        for level in self.levels:
            if not self.layers[level].is_model() or self.layers[level].model is not None:
                continue
            print("train:", str(self.layers[level]), "for", level.__name__, "layer.")
            self.layers[level].train(self.trainingset)
            print("Done.")
            self.layers[level].save()

