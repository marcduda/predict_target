from sklearn.pipeline import make_pipeline

from feature_transformers import SplitFeaturesTargetTransformer, ConstantValueImputer,\
    ToTimestampTransformer, ChangeTypeToInt, KeepColumnsTransformer

pipeline = make_pipeline(
    ToTimestampTransformer(),
    ConstantValueImputer(),
    ChangeTypeToInt(),
    KeepColumnsTransformer(),
    SplitFeaturesTargetTransformer(),
)
