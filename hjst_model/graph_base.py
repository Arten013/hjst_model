from components.sql.sql_deco import execute, executemany, set_tracer
import re
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx
import graphviz as viz
from copy import copy
from itertools import combinations, groupby
from operator import itemgetter
import random
from collections import OrderedDict
import math
from pprint import pprint

# 本文の一行あたり文字数
NLWIDTH = 30

import graphviz as viz
def jst2viz(self, elem, bottom_etypes = [], bottom_nodes=[], _viz=None):
    bottoms = [et.__name__ if isisntance(et, type) else et for et in bottom_etypes]
    G = viz or viz.Digraph(self.tag, filename="hoge")
    with G.subgraph(name='cluster_'+str(self.tag)) as sg:
        for child in elem.children:
            if child.etype.__name__ in bottom_etypes or len(child.children) == 0:
                sg.node(child.code, label = str(child))
                if bottom_nodes:
                    sg.edge(bottom_nodes[-1].code, child.code)
                bottom_nodes.append(child)
            else:
                elem2viz(child, bottom_etypes, bottom_nodes, sg)
        sg.node_attr.update(style='filled')
        sg.attr(label=str(elem))
        sg.attr(color='blue')
    return G
# エッジの端点ノードタグstag, etagについてイテレーション
for node in self.node_dict():














def elem_to_node(self, elem, parent_elem=None):
    elem = ElementNode(
            tag = elem.code,
            label = str(elem),
            parent_tag = None,
            sentence = elem.text
    )
    return elem

def jstatutree_to_graph(self):
    pass

class NodeBase(object):
    
# ノードクラスの基底クラス
class ElementNode(object):
    def __init__(self, elem):
        self.tag = elem.code
        self.label = str(elem)
        self.etype = elem.etype
        
        # sentenceをNLWIDTH文字ごとに改行
        sentence = re.sub(r"\s", "", elem.text) + (" " * NLWIDTH)
        self.sentence = "\n".join([sentence[i:i+NLWIDTH] for i in range(0, len(sentence)-NLWIDTH, NLWIDTH)])
        
        self.subgraph = GraphBase(self.tag)
        
    def set_children(self, children):
        assert len(self.subgraph.node_dict) == 0, 'You cannot set child nodes multiple times.'
        children = list(children)
        if not isinstance(children[0], ElementNode):
            assert isinstance(children[0], jstatutree.TreeElement), 'You must pass TreeElement objects for ElementNode.set_children.'
            children = [ElementNode(c) for c in children]
        self.subgraph.set(self.label, self.tag, children)

    # networkxの関数に代入するときに使用する関数
    def nxnode(self):
        # クラス変数をコピー
        ret_dict = copy(self.__dict__)

        # タグのkeyをnetworkx用に合わせて変更
        del ret_dict["tag"]
        ret_dict["node_for_adding"] = self.tag

        return ret_dict
        
# グラフの基底クラス
class ElementGraph(nx.DiGraph):
	BASE_COLOR = "blue"
	def __init__(self, tag, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# ノードを記録する順序付き辞書
		self.node_dict = OrderedDict()
		self.tag = str(tag)

	# 基本情報の設定
	# 基本となるグラフも生成
	def set(self, name, tag, nodes):
		self.name = str(name)
		self.tag = str(tag)

		# ノードに起点と終点を設定
		nodes = [StartNode(self_tag)] + nodes + [EndNode(self_tag)]

		# ノードを順番に結合
		for node in nodes:
			self.add_node(**node.nxnode())

		# add edges
		self.add_path(self.node_dict.keys(), weight=1)

	# 先頭ノードを取り出し
	# 起点ノードを取り出すことも可能
	def head(self, start=True):
		i = 0 if start else 1
		return list(self.node_dict.items())[i]

	# 末尾ノードを取り出し
	# 終点ノードを取り出すことも可能
	def tail(self, end=True):
		i = -1 if end else -2
		return list(self.node_dict.items())[i]

	# ノードの追加 (networkx.DiGraphのものを上書き)
	def add_node(self, node_for_adding, attr_dict=None, **attr):
		if node_for_adding in self.node_dict.keys():
			return
		super().add_node(node_for_adding=node_for_adding, attr_dict=attr_dict, **attr)
		
		# 辞書として追加された引数が空だった場合の処理
		attr_dict = {} if attr_dict is None else attr_dict
		attr = {} if attr is None else attr

		self.node_dict[node_for_adding] = ElementNode(tag=node_for_adding, **attr_dict, **attr)

	# graphviz形式のグラフを返す関数
	def graphviz(self, _viz=None):
		G = viz or viz.Digraph(self.tag, filename="hoge")
		with G.subgraph(name='cluster_'+str(self.tag)) as c:
			# エッジの端点ノードタグstag, etagについてイテレーション
            for node in self.node_dict():
                

			# ノードの見栄えを整える
			c.node_attr.update(style='filled')
			c.attr(label=self.name)
			c.attr(color=self.BASE_COLOR)
		return G

    
    
# 条グラフ
class ArticleGraph(GraphBase):
	pass

class OrdinanceGraph(GraphBase):
	pass