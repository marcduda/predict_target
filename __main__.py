import numpy as np
import random
from datetime import datetime
import pandas as pd
import logging
import joblib
import os

from model import CatBoostTransformer
from argparser import project_argparser
from pipelines import pipeline
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.set_printoptions(threshold=np.inf)

np.random.seed(seed=0)
random.seed(10002)


def export_pickle(pickle_object, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(pickle_object, path, 5)


if __name__ == '__main__':
    args = project_argparser.parse_args()
    train_full_model = args.train_full_model
    out_directory = args.out_directory
    input_file = args.input_file
    previous_accuracy = args.previous_accuracy
    model_version = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S-UTC")
    predict = args.predict
    test_file = args.test_file
    model_file = args.model_file

    preproc_pipeline = pipeline
    # Model
    model = CatBoostTransformer(preproc_pipeline)

    if train_full_model:
        df_raw = pd.read_csv(input_file)

        logger.info("Shape before preprocessing: {}\nColumns: {}".format(df_raw.shape, df_raw.columns))
        features, target = preproc_pipeline.transform(df_raw)
        logger.info("Shape after preprocessing: {}\nColumns: {}".format(features.shape, features.columns))

        model.fit(features, target)
        out_path = os.path.join(out_directory, "model.joblib")
        logger.info("Model fitted. Exporting to {}".format(out_path))
        if previous_accuracy is not None:
            #TODO find a better way to evaluate the accuracy
            y_pred = model.predict(df_raw)
            current_accuracy = accuracy_score(y_pred, target)
            if float(previous_accuracy) <= current_accuracy:
                export_pickle(model, out_path)
            else:
                logger.info("The new trained model didn't performed as good as a previous one so it is not saved")
    if predict:
        if test_file is not None:
            if model_file:
                model_test = joblib.load(model_file)
                df_test = pd.read_csv(input_file)
                y_pred = model_test.predict(df_test)
                y_pred.to_csv('predictions_'+datetime.utcnow().strftime("%Y-%m-%d-%H%M%S-UTC")+'.csv', index=False)
            elif model:
                df_test = pd.read_csv(input_file)
                y_pred = model.predict(df_test)
                y_pred.to_csv('predictions_' + datetime.utcnow().strftime("%Y-%m-%d-%H%M%S-UTC") + '.csv', index=False)
            else:
                logger.info("Trying to do a prediction on a data file but without a current or saved model")
        else:
            logger.info("Trying to do a prediction without test data")

