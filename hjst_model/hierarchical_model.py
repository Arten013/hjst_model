import itertools
import numpy as np
from time import time
import gc
from .scored_pair import ScoredPairGDB
import concurrent
import pandas as pd
from queue import Queue

class HierarchicalModel(object):
    def __init__(self):
        self.layers = dict()
        self.thresholds = dict()
        self.levels = []

    def set_trained_model_layer(self, level, model, threshold):
        assert level not in self.levels, "Level " + str(level)+ " has already exists"
        self.levels.append(level)
        self.layers[level] = model
        self.thresholds[level] = threshold

    def get_layered_compare(self, testset, r1, r2):
        def get_tree_paths(tree):
            paths = []
            edge_queue = Queue()
            print(tree)
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
            return paths
        paths_1, paths_2 = get_tree_paths(r1), get_tree_paths(r2)
        tr_paths_1, tr_paths_2 = np.array(paths_1).transpose(), np.array(paths_2).transpose()
        return np.array(
                [layer.compare_by_idvectors(tr_paths_1[li], tr_paths_2[li])
                for li, layer in enumerate(self.layers.values())]
            )

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

