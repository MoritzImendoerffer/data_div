import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import copy
from sklearn.preprocessing import PolynomialFeatures


class VarSelect:

    """
    1. stores a copy of design_df and factor_info
    2. creates dummy coded columns based on design_df and factor_info
    3. Updates design_df and factor_info with dummy coded columns, delete original columns, preserves order of columns
    4. Creates sklearn polynomial features to make a model of desired depth and stores it in self._Xp
    5. Creates a NetworkX graph to keep track of which interaction consists of which main effects
    7. Provides a view on self._Xp which ensures the heredity principle. Active effects are stored in self._factor_dict
    8. Provides functionality to add or remove effects from the view on self._Xp
    """
    def __init__(self, design_df, factor_info, use_str_names=True, init_full_model=False):
        # dictionaries have to be in order. Test that Python is > 3.5
        assert sys.version >= '3.6', 'Python greater 3.6 required for ordered dicts'

        self._design_df = design_df.copy()
        # {name: dict(type=, effects=)}, original info passed during init. Used for creating dummy coding
        # not supposed to the be used after the dummy coded df is constructed
        self._factor_info = copy.copy(factor_info)
        # will contain the info of which blocking column stands for which name in self._factor_info and self.X
        self._block_info = None

        self._add_dummies(dtypes=('cat', 'bloc'))
        self._poly, self._Xp = self._make_poly_features()
        if use_str_names:
            # set the column names to match the coded feature names
            self._Xp.columns = self._poly.get_feature_names()
        self._G = self._make_graph()

        self._cleanup_xp()
        # this dict contains all factors by column index which should be in the model values will be 1
        self._factor_dict = dict()

    def _add_dummies(self, dtypes=('cat', 'bloc')):
        """
        Adds the dummy coded columns to self._X and stores the information of each datatype in self.factor_info_bloc
        The order in self.factor_info_bloc represents the actual order of the columns in self._X

        :param dtypes: define which dtypes should be processed. Default is blocks and categorical
        :param drop_col: drops the original columns
        :return:
        """
        added_cols = []
        drop_cols = []
        for dtype in dtypes:
            df, cols_to_drop = self._make_dummies(dtype=dtype)
            added_cols.append((df.columns.to_list(), dtype))
            drop_cols.extend(cols_to_drop)
            self._design_df.drop(columns=cols_to_drop, inplace=True)
            self._design_df = self._design_df.join(df)

        for col in drop_cols:
            _ = self._factor_info.pop(col)

        # add the column type to the info dict
        for columns, dtype in added_cols:
            for col in columns:
                self._factor_info[col] = dict(dtype=dtype)

        # keep the block info
        self._block_info = {k: k.split('_')[0] for k, v in self._factor_info.items()}
        #  rename the factor_info keys to match those of the polynomial features
        _clean_keys = {f'x{idx}': item[1] for idx, item in enumerate(self._factor_info.items())}
        # because factor index should map all main factors in the correct order,
        # the intercept ('1') is added at position 0
        _factor_info = {'1': {'dtype': 'cont'}}
        for k, v in _clean_keys.items():
            _factor_info[k] = v
        self._factor_info = _factor_info

    def _cleanup_xp(self):
        """
        Removes interaction with blocking factors,
        quadratic blocking and quadratic categorical factors from self._G and self._Xp
        :return: None
        """
        # get all interactions by index
        interactions = self._get_effects(filter='interaction')

        # get the main factors which make up the interaction
        interactions_items = [self._get_connected_factors(i, filter='main') for i in interactions]

        # get only those interactions which contain a block
        bloc_interactions = [interactions[index] for index, item in enumerate(interactions_items)
                             for effect in item if self._get_factor_info(effect) == 'bloc']

        # get all quadratic factors
        quadratic = self._get_effects(filter='quadratic')

        # get the quadratic factors which are blocks
        bloc_quadratic = [i for i in quadratic if self._get_factor_info(self._get_connected_factors(i)[0]) == 'bloc']
        cat_quadratic = [i for i in quadratic if self._get_factor_info(self._get_connected_factors(i)[0]) == 'cat']

        # remove not allowed factor combinations from the dataframe and the graph
        not_allowed = []
        not_allowed.extend(bloc_interactions)
        not_allowed.extend(bloc_quadratic)
        not_allowed.extend(cat_quadratic)

        cols = self._Xp.columns.to_list()
        for node in set(not_allowed):
            col = cols[node]
            self._Xp.drop(columns=col)
            self._G.remove_node(node)

    def _make_dummies(self, dtype, drop_first=True):
        """
        Uses pd.get_dummies to creaty dummy columns from categorical and blocking features as defined in the dict
        self._factor_info
        :param dtype: 'cat' for categorical or 'bloc' for blocking factors
        :return: dataframe with dummy coded columns (n-1) per default
        """
        # create dummy columns only for the selected categorical or blocking columns
        col_names = [k for k, v in self._factor_info.items() if v['dtype'] == dtype]
        df = pd.get_dummies(self._design_df.loc[:, col_names], drop_first=drop_first, columns=col_names)
        return df, col_names

    def _make_poly_features(self, degree=2):
        poly = PolynomialFeatures(degree).fit(self._design_df)
        df = pd.DataFrame(poly.fit_transform(self._design_df))
        return poly, df

    def _make_graph(self):
        """
        Creates a NetworkX Graph which contains mappings of factors in the polynomial feature matrix
        :return:
        """
        # creates the mapping of the created columns. Quadratic factors will be converted to weights of the edges
        G = nx.from_numpy_array(self._make_adjacency(), parallel_edges=False)

        return G

    def _make_adjacency(self):
        """
        Creates the adjacency matrix of the polynomial feature matrix. This matrix contains all mappings of which
        column became which interaction.
        :return: adjacency matrix
        """
        # fill up with zeros to create an adjacency matrix
        # first column with zeros represents the intercept
        powers = self._poly.powers_
        _p = np.hstack((np.zeros((powers.shape[0], 1)), powers))
        adj = np.hstack((_p,
                         np.zeros((_p.shape[0],
                                   _p.shape[0] -
                                   _p.shape[1]))))
        return adj

    def _plot_graph(self):
        nx.draw(self._G, with_labels=True,
                font_color='k',
                node_color='y',
                connectionstyle='arc3, rad = 0.1')
        plt.show()

    def _get_effects(self, filter: str):
        """
        Returns a list with all interaction pairs
        :return: list of interaction pairs by index
        """
        main = [i for i in self._G.nodes if self._get_effect_type(i) == filter]
        return main

    def _get_factor_info(self, idx, check_if_main=True):
        """
        Returns the type of a main factor from self._factor_info {'cat', 'bloc', 'cont'}
        :param idx: index of the factor
        :return:
        """
        info = [item[1]['dtype'] for i, item in enumerate(self._factor_info.items()) if i == idx]
        if info:
            return info[0]
        else:
            return None

    def _get_effect_type(self, idx):
        """
        Return the type of an effect {'main', 'interaction', 'quadratic'}

        :param idx: the index (0 to N-1 columns) of the polynomial features
        :return: one of {'main', 'interaction', 'quadratic'}
        """
        # the idea is the following: loop over all edges of the node in question
        # check the properties of the edges:

        # main or interaction effects af a weight of 1
        is_main_or_inter = [e[2]['weight'] == 1.0 for e in self._G.edges(idx, data=True)]
        if is_main_or_inter:
            # all returns True for empty lists
            is_main_or_inter = all(is_main_or_inter)
        else:
            is_main_or_inter = False

        # main effects have a connection with each other
        is_main = [True if e[0] == e[1] else False for e in self._G.edges(idx)]
        if is_main:
            is_main = any(is_main)
        else:
            is_main = False
        # quadratic effects do have a weight of 2
        is_quadratic = [e[2]['weight'] == 2.0 for e in self._G.edges(idx, data=True)]
        if is_quadratic:
            is_quadratic = all(is_quadratic)
        else:
            is_quadratic = False

        if is_quadratic:
            return 'quadratic'
        elif is_main:
            return 'main'
        elif is_main_or_inter:
            return 'interaction'
        elif idx == 0:
            return 'intercept'
        else:
            raise TypeError(f'Type of node with index {idx} not known')

    def _get_connected_factors(self, idx, filter: str = 'main'):
        """
        Get all factor filtered by its type which make up the corresponding factor
        :param idx:
        :param filter: one of {'main', 'intercept', 'quadratic'}
        :return: list of node indexes which are connected and main effects
        """
        if self._get_effect_type(idx) == 'intercept':
            return 0
        else:
            return [n for n in self._G.neighbors(idx) if self._get_effect_type(n) == filter]

    def _get_feature_name(self, idx):
        return self._poly.get_feature_names()[idx]

    @property
    def X(self):
        """
        Choses the colunms from self._Xp as defined in self._factor_dict
        :return: view on self._Xp
        """
        return self._Xp.loc[:, self._factor_dict.keys()]

    def _factor_filter(self, filter: str = 'main'):
        """
        Get all nodes of certain type
        :param filter: one of {'main', 'intercept', 'quadratic'}
        :return: list of node indexes which are of type defined by filter
        """
        return [n for n in self._G.nodes if self._get_effect_type(n) == filter]

    def remove_factor(self, idx: int, heredity: bool = True):
        """
        Removes the columns from self.X
        :param idx:
        :param heredity:
        :return:
        """
        cols_to_remove = []
        cols_to_remove.append(idx)
        # remove also the interactions if this is a main effect
        if heredity:
            interactions = self._get_connected_factors(0, filter='main')
            cols_to_remove.extend(interactions)
            
        for k in cols_to_remove:
            # pops the key if it exists, otherwise return None
            _ = self._factor_dict.pop(k, None)

    def add_factor(self, idx, heredity=True):
        """
        Adds a factor selected by idx to the model and ensures the heredity principle if selected
        :param idx: factor to add
        :param heredity:
        :return:
        """
        cols_to_add = []
        cols_to_add.append(idx)

        if heredity:
            # remove also main effects if the factor is an interaction
            mains = self._get_connected_factors(idx, filter='main')
            cols_to_add.extend(mains)
        for k in cols_to_add:
            # add the factor index as dictionary key
            self._factor_dict[k] = 1
