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

    def refine_pairs(self, testset, candidate_pairs):
        for li, level in enumerate(self.levels):
            candidate_num = len(candidate_pairs[level])
            t1 = time()
            self.layers[level].refine_scored_pairs(candidate_pairs[level], threshold=self.thresholds[level])
            t2=time()
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
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
