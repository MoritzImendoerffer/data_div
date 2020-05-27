import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import copy


class VarSelect:

    def __init__(self, design_df, factor_info):
        self._design_df = design_df.copy()
        self._X = design_df.copy()
        # {name: dict(type=, effects=)}
        self._factor_info = copy.copy(factor_info)

        self._add_dummies()

    def _add_dummies(self, dtypes=['cat', 'bloc']):
        """
        Adds the dummy coded columns to self._X and stores the information of each datatype
        :param dtypes: define which dtypes should be processed. Default is blocks and categorical
        :param drop_col: drops the original columns
        :return:
        """
        for dtype in dtypes:
            df, cols_to_drop = self._make_dummies(dtype=dtype)
            for col in df.columns:
                # add the column type to the info dict
                self._factor_info[col] = dict(dtype=dtype)
            self._X = self._X.join(df)
            self._X.drop(columns=cols_to_drop, inplace=True)
            _ = [self._factor_info.pop(c) for c in cols_to_drop]

    def _make_dummies(self, dtype='cat', drop_first=True):
        """

        :param dtype: 'cat' for categorical or 'bloc' for blocking factors
        :return: dataframe with dummy coded columns (n-1) per default
        """
        # create dummy columns only for the selected categorical or blocking columns
        col_names = [k for k, v in self._factor_info.items() if v['dtype'] == dtype]
        df = pd.get_dummies(self._X.loc[:, col_names], drop_first=drop_first, columns=col_names)
        return df, col_names




