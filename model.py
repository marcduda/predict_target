import logging
import os
import time

import numpy as np
from sklearn.base import BaseEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(seed=0)


class CatBoostTransformer(BaseEstimator):
    def __init__(self, pipe_feature_engineering):
        self.pipe_feature_engineering = pipe_feature_engineering

    def fit(self, df, y=None, verbose=True):
        logger.info("Apply feature engineering pipe to training dataframe...")
        feature = self.pipe_feature_engineering.fit_transform(df, y)
        feature = feature.tocsc()  
        logger.info("Input shape: %s", feature.shape)
        logger.info("Training model %s ...", self.name)
        target = y
        start = time.time()
        self.model.fit(feature, target, verbose=verbose)
        logger.info("Training model took %s seconds", time.time() - start)
        return self.model

    def predict(self, df):
        logger.debug("Apply feature engineering pipe to dataframe...")
        feature = self.pipe_feature_engineering.transform(df)
        logger.debug("Predict from model...")
        predictions = self.model.predict(feature)
        return predictions.tolist()



