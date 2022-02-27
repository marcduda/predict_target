import numpy as np
import random
from datetime import datetime
import pandas as pd
import logging

from model import CatBoostTransformer
from argparser import project_argparser
from pipelines import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.set_printoptions(threshold=np.inf)

np.random.seed(seed=0)
random.seed(10002)


if __name__ == '__main__':
    args = project_argparser.parse_args()
    #cross_val = args.cross_val
    train_full_model = args.train_full_model
    out_directory = args.out_directory
    input_file = args.input_file
    previous_accuracy = args.previous_accuracy
    model_version = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S-UTC")

    preproc_pipeline = pipeline
    df_raw = pd.read_csv(input_file)

    logger.info("Shape before preprocessing: {}\nColumns: {}".format(df_raw.shape, df_raw.columns))
    features, target = preproc_pipeline.transform(df_raw)
    logger.info("Shape after preprocessing: {}\nColumns: {}".format(features.shape, features.columns))

    # Model
    model = XGBTransformer(preproc_pipeline)
    
    '''
    if cross_val:
        logger.info("Computing cross val model predictions...")
        performances, df_with_predictions, train_index, test_index = model.perform_cross_val(features, target, cv=4, verbose=True)

        # Export to out_directory locally and on Cloud storage
        with open(out_directory+'/'+'model_performance_'+ model_version + '.json', 'w') as f:
            json.dump(performances, f)
        with open(out_directory+'/'+'config_feature_engineering_' + model_version + '.json', 'w') as f:
            json.dump(config_feature_engineering, f)
        with open(out_directory+'/'+'config_model_' + model_version + '.json', 'w') as f:
            json.dump(config_model, f)
        df_with_predictions.to_csv(out_directory+'/'+'data_with_pred_' + model_version + '.csv', index=False)
    '''
    if train_full_model:
        model.fit(features, target)
        out_path = os.path.join(out_directory, "model.joblib")
        logger.info("Model fitted. Exporting to {}".format(out_path))
        export_pickle(model, out_path)
    if predict:

