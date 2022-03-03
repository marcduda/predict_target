import logging

import pandas as pd
from sklearn.base import TransformerMixin
from config import *
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MakeCopyTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        """
        :param df: Dataframe to be copied
        :return: copied Dataframe
        """
        return df.copy()


class ToTimestampTransformer(TransformerMixin):
    def __init__(self, cols=time_features):
        self.columns = cols

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        """
        change the values of selected columns to datetime format
        :param df: Dataframe on which to apply changes
        :return: Dataframe with the columns changed
        """
        for c in self.columns:
            if c in list(df.columns):
                df[c] = df[c].apply(lambda x: datetime.strptime(str(int(x)), '%Y%m%d') if not pd.isna(x) else x)
        return df


class ConstantValueImputer(TransformerMixin):
    def __init__(self, dict_values=dict_values):

        self.dict_values = dict_values

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        """
        fill the missing values with a constant
        :param df: Dataframe on which to apply changes
        :return: Dataframe with missing values imputed
        """
        for features_group, value in self.dict_values.values():
            df[features_group] = df[features_group].fillna(value)
        return df


class ChangeTypeToInt(TransformerMixin):
    def __init__(self, cols=time_features):
        self.columns = cols

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        """
        change the values of selected columns to int format
        :param df: Dataframe on which to apply changes
        :return: Dataframe with the columns changed
        """
        df[self.columns] = df[self.columns].astype('int64') // 10 ** 9 // 100
        return df


class KeepColumnsTransformer(TransformerMixin):
    def __init__(self, columns=columns_to_keep):
        self.columns_to_keep = columns

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        """
        select a sub-Dataframe with only some columns
        :param df: Dataframe on which to apply changes
        :return: Dataframe with only some columns kept
        """
        return df[self.columns_to_keep]


class SplitFeaturesTargetTransformer(TransformerMixin):
    def __init__(self, target=target_column):
        self.target_column = target

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        """
        split the Dataframe into a Dataframe of features and a Series for the target
        :param df: Dataframe on which to apply changes
        :return: a tuple of (feature as a Dataframe, target as Series)
        """
        target = df[self.target_column]
        features = df.drop(self.target_column, 1)
        return features, target
