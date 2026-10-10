"""Pipelines MLlib de feature engineering (indexation, encodage, assemblage, standardisation)."""

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

NUMERIC_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]


def build_classification_pipeline(categorical_cols=CATEGORICAL_COLS, numeric_cols=NUMERIC_COLS, label_col="income"):
    """Construit le pipeline MLlib : StringIndexer -> OneHotEncoder -> VectorAssembler -> label indexing."""
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in categorical_cols]
    encoder = OneHotEncoder(
        inputCols=[c + "_index" for c in categorical_cols],
        outputCols=[c + "_encoded" for c in categorical_cols],
    )
    assembler = VectorAssembler(
        inputCols=numeric_cols + [c + "_encoded" for c in categorical_cols],
        outputCol="features",
    )
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label")

    return Pipeline(stages=indexers + [encoder, assembler, label_indexer])


def split_train_test(df, train_ratio=0.8, seed=2025):
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)


def fit_transform_pipeline(pipeline, train_data, test_data):
    model = pipeline.fit(train_data)
    return model, model.transform(train_data), model.transform(test_data)


def scale_features(train_transformed, test_transformed, input_col="features", output_col="scaledFeatures"):
    """Standardise les features de classification (StandardScaler fit sur le train)."""
    scaler = StandardScaler(inputCol=input_col, outputCol=output_col)
    scaler_model = scaler.fit(train_transformed)
    return scaler_model.transform(train_transformed), scaler_model.transform(test_transformed)


def build_clustering_pipeline(numeric_cols=NUMERIC_COLS):
    """Construit le pipeline de préparation des features numériques pour le clustering."""
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")
    return Pipeline(stages=[assembler, scaler])
