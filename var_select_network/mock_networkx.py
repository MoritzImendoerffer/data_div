"""
Trying networkX
"""

import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

tree = (('B1', 'B2'), ('B2', 'B3'), ('B3', 'B4'),
        ('B3', 'B5'), ('B5', 'B6'),
        ('B4', 'B6'), ('B6', 'B7'),
        ('B7', 'B9'), ('B6', 'B8'),
        ('B8', 'B9'), ('B9', 'B10'),
        ('B4', 'B11'), ('B4', 'B12'))

gr = nx.DiGraph()

gr.add_edges_from(tree)
gr_tree = nx.dfs_tree(gr)

fig, ax = plt.subplots(ncols=1)
nx.draw(gr, ax=ax, with_labels=True, font_size=20, font_color='red',
        node_color='white')
plt.show()

for i, n in enumerate(gr.nodes):
    gr.add_node(n, ind=i)

gr.nodes.data()

data = nx.get_node_attributes(gr, 'ind')
df = pd.DataFrame.from_dict(data, orient='index')


[list(gr.successors(n)) for n in gr.nodes]

gr.graph_attr_dict_factory()

adj = nx.adjacency_matrix(gr)