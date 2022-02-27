import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(seed=0)


class CatBoostTransformer(BaseEstimator):
    def __init__(self, pipe_feature_engineering):
        self.pipe_feature_engineering = pipe_feature_engineering
        self.model = CatBoostClassifier(
            iterations=2,
            verbose=1,
        )

    def fit(self, feature, target=None, verbose=True):
        logger.info("Apply feature engineering pipe to training dataframe...")
        start = time.time()
        self.model.fit(feature, target, verbose=verbose)
        logger.info("Training model took %s seconds", time.time() - start)
        return self.model

    def predict(self, df):
        logger.debug("Apply feature engineering pipe to dataframe...")
        feature, _ = self.pipe_feature_engineering.transform(df)
        logger.debug("Predict from model...")
        predictions = self.model.predict(feature)
        predictions = pd.DataFrame(data=predictions, columns=['predictions'])
        return predictions



