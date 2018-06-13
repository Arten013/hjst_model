import itertools

class HierarchicalModel(object):
    def __init__(self, trainingset):
        self.trainingset = trainingset
        self.levels = trainingset.levels
        self.models = dict()
        self.thresholds = dict()
        self.top_rates = dict()

    def set_model(self, level, model_cls, savepath, threshold=0.3, top_rate=0.7):
        assert level in self.levels, "This level is not in levels"+str(levels)
        try:
            self.models[level] = model_cls.load(savepath)
            print("Load model:", savepath)
        except:
            self.models[level] = model_cls(level, savepath)
            print("Create model.")
        self.thresholds[level] = threshold
        self.top_rates[level] = top_rate

    def batch_training(self):
        assert len(self.models) == len(self.levels), "You must register models for all levels."
        for level in self.levels:
            if self.models[level].model is not None:
                continue
            self.models[level].train(self.trainingset)
            self.models[level].save()

    
    def refine_pairs(self, candidate_pairs):
        for li, level in enumerate(self.levels):
            t1 = time()
            candidate_num = len(candidate_pairs)
            scored_pairs = [None]*candidate_num
            si = 0
            for k1, k2 in candidate_pairs:                 
                score = self.models[level].compare(k1, k2)
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

    def refine_pairs(self, candidate_pairs):
        for li, level in enumerate(self.levels):
            candidate_num = len(candidate_pairs[level])
            si = 0
            t1 = time()
            for k1, k2 in candidate_pairs[level]:                 
                score = self.models[level].compare(k1, k2)
                if self.thresholds[level] is None or score >= self.thresholds[level]:
                    candidate_pairs[level].add_scored_pair(k1, k2, score)
                    si += 1
                else:
                    candidate_pairs[level].del_pair(k1, k2)
            print("threshold pass: {0:,}/{2:,}".format(si, candidate_num))
            t2=time()
            print("refining_time({}):".format(level.__name__), t2-t1, "sec")
            if level == self.levels[-1]:
                return scored_pairs
            candidate_pairs[level].add_by_iterable_pairs(
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

    def get_pairs(self, testset, dbpath):
        candidate_pairs = LeveledScoredPairKVS(path=dbpath, levels=self.levels)
        candidate_pairs.add_by_iterable(itertools.combinations(testset.sentence_dict[self.levels[0]].keys(), 2))
        return self.refine_pairs(candidate_pairs)