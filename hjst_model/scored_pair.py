from jstatutree.kvsdict import KVSDict

import os
import itertools
import re

class ScoredPairKVS(KVSDict):
    DEFAULT_DBNAME = "ScoredPair.ldb"
    ENCODING = "utf8"
    PREFIX = "example-"

    def add_by_iterable(self, list):
        self.add_by_iterable_pairs(itertools.combinations(iterable, 2))

    def add_by_iterable_pairs(self, iterable):
        for i1, i2 in iterable:
            self.add_scored_pair(i1, i2, 0)

    def add_scored_pair(self, i1, i2, score):
        key = "-".join(sorted([i1, i2]))
        self[key] = score

    def __iter__(self):
        for key, score in self.items():
            former, latter = re.split("-", key)
            yield former, latter, score

    def del_pair(self, i1, i2):
        key = "-".join(sorted([i1, i2]))
        del self[key]

class LeveledScoredPairKVS(object):
    def __init__(self, path, levels):
        self.levels = levels
        self.level_dict = {
            l:ScoredPairKVS(path=os.path.join(path, "ScoredPairs-{}.ldb").format(l.__name__))
            for l in self.levels
            }

    def __getitem__(self, key):
        return self.level_dict[key]