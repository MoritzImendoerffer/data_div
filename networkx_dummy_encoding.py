"""
This is a test whether NetworkX might be suited for feature selection in the
context of categorical and blocking factors

Requirements

no interactions or quadratic effects with blocking factors
interactions but no quadratic factors for categorical variables
interactions and any order for continuous factors

remove a categorical factor and all of its interactions

Idea

Create polynomial features (easy)
Use the poly._powers matrix to
"""


from sklearn.preprocessing import PolynomialFeatures
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from VarSelect import VarSelect

X = np.random.random((3, 5))
X = pd.DataFrame(X)
X.columns = [f'x{i}' for i in range(X.shape[1])]
factor_info = {k: dict(dtype='cont') for k in X.columns}
factor_info['x3']['dtype'] = 'cat'
factor_info['x4']['dtype'] = 'bloc'

vs = VarSelect(design_df=X, factor_info=factor_info)
vs._plot_graph()

"""
Code below are snippets for later usage
"""
poly = PolynomialFeatures(2).fit(X)
Xp = poly.fit_transform(X)
df = pd.DataFrame(Xp)
powers = poly.powers_
names = ['x0', 'x1', 'x2']
pnames = poly.get_feature_names()
# fill up with zeros to create an adjacency matrix
# first column with zeros represents the intercept
_p = np.hstack((np.zeros((powers.shape[0], 1)), powers))
adj = np.hstack((_p,
                 np.zeros((_p.shape[0],
                           _p.shape[0] -
                           _p.shape[1]))))

G = nx.from_numpy_array(adj, parallel_edges=False)
nx.draw(G, with_labels=True,
                 font_color='k',
                 node_color='y',
                 connectionstyle='arc3, rad = 0.1')
plt.show()
# mapping the factor index (name of the nodes)
# with the string from polynomial featuresEW
name_map = {n: i for i, n in enumerate(names)}

# removing all item containing a factor seems easy
# there is a quirk in the network
def remove_factor(name, main=False, inter=False, quadr=False):
    """
    Removes all columns which contain a specific name
    :param name:
    :return:
    """
    idx = name_map[name]
    # the index of all columns containing the factor
    f_idx = [n for n in G.neighbors(idx)]
    # get all other columns
    mask = df.columns.difference(f_idx)
    print(df.loc[:, mask])


remove_factor('x2')

# removing only the interactions of a factor
idx = 1  # graph node
f_idx = [n for n in G.neighbors(idx)]  # neighbors+
f_idx
# remove main effect would mean, remove all f_idx
# remove interactions and quadratic would mean remove f_idx[1:]

# interactions are those edges with weight 1
interactions = [e[1] for e in G.edges(idx, data=True) if e[2]['weight'] == 1][1:]
interactions

# quadradic factors are those interactions with weight 2
quadratic = [e[1] for e in G.edges(idx, data=True) if e[2]['weight'] == 2]
quadratic

# check if is interaction or main effect
idx = 1
all([e[2]['weight'] == 1.0 for e in G.edges(idx, data=True)])

# check if it is a main effect
# each main effect has an edge with itself
idx = 2
any([True if e[0] == e[1] else False for e in G.edges(idx)])
# check if is main effect

# is quadratic?
idx = 1
all([e[2]['weight'] == 2.0 for e in G.edges(idx, data=True)])


'''
The code above would enable to:

1. create a full matrix using polynomial features
2. remove all interactions and quadratic factors for blocs
3. remove all quadratic factors for a categorical factor
4. remove main effects, interactions or quadratic factors
   for any factor (mutually inclusive)
   
What about the opposite ?
Is it possible to include certin factors and their combinations?
If the function returns a view on the dataframe,
it should be possible to do that. 
'''




