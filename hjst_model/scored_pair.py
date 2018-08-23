from jstatutree.kvsdict import KVSDict

import os
import itertools
import re
import numpy as np
from neo4j.v1 import GraphDatabase
from threading import Thread
from queue import Queue

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

    def get_score(self, i1, i2, default=None):
        key = "-".join(sorted([i1, i2]))
        return self.get(key, default)

    def __iter__(self):
        for key, score in self.items():
            former, latter = re.split("-", key)
            yield former, latter, score

    def del_pair(self, i1, i2):
        key = "-".join(sorted([i1, i2]))
        del self[key]

class ScoredPairGDB(object):
    def __init__(self, loginkey, experiment_name, level, dataset_name):
        self.driver = GraphDatabase.driver(**loginkey)
        self.name = experiment_name
        self.level_label = (level.__name__ if isinstance(level, type) else level) + 's'
        self.dataset_name = dataset_name
        self.pair_queue = Queue(10000)

    def _enqueue_pairs(self, vecs):
        target = self.get_candidate_gdbwva_pairs if vecs else self.get_candidate_pairs
        while True:
            with self.driver.session() as session:
                pairs = session.write_transaction(target)
            if len(pairs) == 0:
                self.pair_queue.put(None)
                break
            for pair in pairs:
                self.pair_queue.put(pair)

    def iter_candidate_pairs(self, vecs=False):
        th = Thread(target=self._enqueue_pairs, args=(vecs,), daemon=True)
        th.start()
        while True:
            item = self.pair_queue.get()
            if item is None:
                raise StopIteration
            yield item

    def init_candidate_edge(self, limit=None):
        with self.driver.session() as session:
            count = 0
            session.run(
                        """
                            MATCH (:%s)-[:MUNI_OF]->(sm)-[:HAS_ELEM]->(m)
                            WITH m
                            %s
                            SET m.%s = true
                        """ % (
                                self.dataset_name,
                                '' if limit is None else 'LIMIT %d' % limit,
                                self.name
                            )
                        )
            while True:
                n = session.run("""
MATCH (:%s)-[:MUNI_OF]->()-[:HAS_ELEM]->(n:Elements{%s: true})
WHERE NOT ((n)-[:%s]->())
with n
limit 1
MATCH (m:Elements{%s: true})
WHERE NOT n = m
AND NOT ((n)-[:%s]-(m))
WITH n, m
CREATE (n)-[:%s{score: 0, yielded: false}]->(m)
RETURN COUNT(n)
                        """ % (
                            self.name,
                            self.name,
                            self.name,
                            self.name,
                            self.name
                            )
                        ).single()[0]
                if n == 0:
                    return count
                count += n

    def get_candidate_pairs(self, tx, size=10000):
        return tx.run("""
                        MATCH (n:%s)-[ce:%s{score: 0, yielded: false}]->(m)
                        WITH n, m, ce
                        LIMIT %d
                        SET ce.yielded = true
                        RETURN n.id, m.id
                    """ % (self.level_label, self.name, size)).values()

    def get_candidate_pairs_with_vecs(self, tx, size=10000):
        return tx.run("""
                        MATCH (n:%s)-[ce:%s{score: 0, yielded: false}]->(m)
                        WITH n, m, ce
                        LIMIT %d
                        SET ce.yielded = true
                        RETURN n.id, n.wva_vec, m.id, m.wva_vec
                    """ % (self.level_label, self.name, size)).values()

    def add_not_passed_pair(self, i1, i2, score):
        with self.driver.session() as session:
            return session.run("""
                MATCH (n:%s{id: '%s'})-[ce:%s{score: 0, yielded: true}]->(m{id: '%s'})
                SET ce.score = %s
                REMOVE ce.yielded
                """ % (
                        self.level_label, i1, self.name, i2, str(round(score, 3))
                    )
                ).values()
    def add_passed_pair(self, i1, i2, score):
        with self.driver.session() as session:
            return session.run("""
                MATCH (n:%s{id: '%s'})-[ce:%s{score: 0, yielded: true}]->(m{id: '%s'})
                SET ce.score = %s
                REMOVE ce.yielded
                WITH n, m
                MATCH (nc)<-[:HAS_ELEM]-(n), (m)-[:HAS_ELEM]->(mc)
                CREATE (nc)-[:%s{score: 0, yielded: false}]->(mc)
                return nc.id, mc.id
                """ % (
                        self.level_label, i1, self.name, i2, str(round(score, 3)), self.name
                    )
                ).values()

class LeveledScoredPairKVS(object):
    def __init__(self, path, levels):
        self.levels = [l if isinstance(l, str) else l.__name__ for l in levels]
        self.level_dict = {
            l:ScoredPairKVS(path=os.path.join(path, "ScoredPairs-{}.ldb").format(l))
            for l in self.levels
            }

    def __getitem__(self, key):
        return self.level_dict[key]

    def get_score_csv(self, path, dataset, model, s1, s2):
        tables = list(self.get_score_table(dataset, model, s1, s2))
        if None in tables:
            return
        for i, table in enumerate(tables):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, self.levels[i]+'.csv'), 'w') as f:
                for row in table:
                    f.write(', '.join([str(r) for r in row])+'\n')

    def get_score_table(self, dataset, model, s1, s2):
        elems_y, elems_x = [dataset.kvsdicts['edges']['Statutory'][s] for s in [s1, s2]]
        for level in self.levels:
            table = [['']+[dataset.kvsdicts['texts'][level][ex] for ex in elems_x]]
            yield_flag = False
            for ey in elems_y:
                table.append([dataset.kvsdicts['texts'][level][ey]])
                for ex in elems_x:
                    score = self[level].get_score(ex, ey)
                    if score is None:
                        score = round((-1)*abs(float(model.layers[level].compare(ex, ey))), 3)
                    else:
                        score = round(abs(float(score)), 3)
                    if abs(score) >= 0.8:
                        yield_flag = True
                    table[-1].append(score)
            if yield_flag:
                yield table
            else:
                yield None
            if level == self.levels[-1]:
                break
            elems_x = [nex for ex in elems_x for nex in dataset.kvsdicts['edges'][level][ex]]
            elems_y = [ney for ey in elems_y for ney in dataset.kvsdicts['edges'][level][ey]]


