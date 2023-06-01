# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:02:54 2023

@author: cbsj0
"""

import networkx as nx
import matplotlib as plt
import numpy as np

G = nx.Graph()
DG = nx.DiGraph()
MG = nx.MultiGraph()
MDG = nx.MultiDiGraph()

#edge_list = [(1,2),(3,4), (1,3), (2,4), (4,5)]
#G.add_edges_from(edge_list)
g= nx.from_numpy_array(np.array([[0,1,0],
                                [1,1,1],
                                [0,0,0]))

nx.draw_networkx(g, with_labels=True)
plt.show()