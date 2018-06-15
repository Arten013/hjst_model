import itertools
from time import time
import gc

class HierarchicalModel(object):
    def __init__(self, trainingset):
        self.trainingset = trainingset
        self.levels = trainingset.levels
        self.layers = dict()
        self.thresholds = dict()

    def set_layer(self, level, layer_cls, savepath, threshold=0.3):
        assert level in self.levels, "This level is not in levels"+str(levels)
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
            self.layers[level].train(self.trainingset)
            self.layers[level].save()

    
    def refine_pairs(self, testset, candidate_pairs):
        for li, level in enumerate(self.levels):
            t1 = time()
            candidate_num = len(candidate_pairs)
            scored_pairs = [None]*candidate_num
            si = 0
            for k1, k2 in candidate_pairs:                 
                score = self.layers[level].compare(k1, k2)
                if self.thresholds[level] is None or score >= self.thresholds[level]:
                    scored_pairs[si] = (k1, k2, score)
                    si += 1
            topn = int(candidate_num * self.top_rates[level])
            print("threshold pass: {0:,}/{2:,}, top_rate pass: {1:,}/{2:,}".format(si, topn, candidate_num))
            if si < topn:
                scored_pairs = scored_pairs[:si]
            else:
                scored_pairs = sorted(scored_pairs[:si], key=lambda x: -x[2])[:topn]
            t2=time()
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            if level == self.levels[-1]:
                return scored_pairs
            candidate_pairs = list(
                itertools.chain.from_iterable(
                        [
                            itertools.product(
                                    testset.statutree_dict[k1],
                                    testset.statutree_dict[k2] 
                                )
                            for k1, k2, _ in scored_pairs
                        ] 
                    )
                )
            print("calculating next pairs:", time()-t2, "sec")
        
        raise("Unexpected error")

    def refine_pairs(self, testset, candidate_pairs):
        for li, level in enumerate(self.levels):
            candidate_num = len(candidate_pairs[level])
            si = 0
            t1 = time()
            for k1, k2, _ in candidate_pairs[level]:                 
                score = self.layers[level].compare(k1, k2)
                if self.thresholds[level] is None or score >= self.thresholds[level]:
                    candidate_pairs[level].add_scored_pair(k1, k2, score)
                    si += 1
                else:
                    candidate_pairs[level].del_pair(k1, k2)
                #del k1, k2, score
                #gc.collect()
                #print(gc.get_objects())
            print("threshold pass: {0:,}/{1:,}".format(si, candidate_num))
            t2=time()
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            if level == self.levels[-1]:
                return candidate_pairs[level]
            for k1, k2, _ in candidate_pairs[level]:
                candidate_pairs[self.levels[li+1]].add_by_iterable_pairs(
                    itertools.product(
                            testset.statutree_dict[k1],
                            testset.statutree_dict[k2] 
                        ) 
                )
            print("calculating next pairs:", time()-t2, "sec")
        
        raise("Unexpected error")

    def get_pairs(self, testset, dbpath):
        candidate_pairs = LeveledScoredPairKVS(path=dbpath, levels=self.levels)
        candidate_pairs.add_by_iterable(itertools.combinations(testset.sentence_dict[self.levels[0]].keys(), 2))
        return self.refine_pairs(testset, candidate_pairs)