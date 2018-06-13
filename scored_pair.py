from xml_jstatutree.jstatutree.kvsdict import KVSDict, KVSPrefixDict

import os
import itertools


class ScoredPairKVS(KVSDict):
    DEFAULT_DBNAME = "ScoredPair.ldb"
    ENCODING = "utf8"
    PREFIX = "example-"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_dicts = dict()

    def add_by_iterable(self, list):
        self.add_by_iterable_pairs(itertools.combinations(iterable, 2))

    def add_by_iterable_pairs(self, iterable):
        for i1, i2 in iterable:
            self.add_scored_pair(i1, i2, 0)

    def add_scored_pair(self, i1, i2, score):
        i1, i2 = sorted([i1, i2])
        if not i1 in self.prefix_dicts:
            self.prefix_dicts[i1] = KVSPrefixDict(self.db, prefix="{}-".format(i1))
        self.prefix_dicts[i1][i2] = score

    def __iter__(self):
        for former in self.prefix_dicts.keys():
            for latter, score in self.prefix_dicts[former].items():
                yield former, latter, score

    def del_pair(self, i1, i2):
        i1, i2 = sorted([i1, i2])
        del self.prefix_dicts[i1][i2]

class LeveledScoredPairKVS(object):
    def __init__(self, path, levels):
        self.levels = levels
        self.level_dict = {
            l:ScoredPairKVS(path=os.path.join(path, "ScoredPairs-{}.ldb").format(l.__name__))
            for l in self.levels
            }

    def __getitem__(self, key):
        return self.level_dict[key]