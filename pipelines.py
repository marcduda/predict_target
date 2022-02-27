from sklearn.pipeline import make_pipeline

from feature_transformers import SplitFeaturesTargetTransformer, ConstantValueImputer,\
    ToTimestampTransformer, ChangeTypeToInt, KeepColumnsTransformer, MakeCopyTransformer

pipeline = make_pipeline(
    MakeCopyTransformer(),
    ToTimestampTransformer(),
    ConstantValueImputer(),
    ChangeTypeToInt(),
    KeepColumnsTransformer(),
    SplitFeaturesTargetTransformer(),
)
