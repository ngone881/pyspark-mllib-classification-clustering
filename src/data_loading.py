"""Téléchargement et chargement du dataset Adult Census Income (UCI)."""

import os

import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income",
]


def get_schema():
    return StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", IntegerType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", IntegerType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", IntegerType(), True),
        StructField("capital_loss", IntegerType(), True),
        StructField("hours_per_week", IntegerType(), True),
        StructField("native_country", StringType(), True),
        StructField("income", StringType(), True),
    ])


def download_dataset(data_dir="."):
    """Télécharge adult.data et adult.test depuis l'UCI s'ils ne sont pas déjà présents."""
    os.makedirs(data_dir, exist_ok=True)
    for filename in ("adult.data", "adult.test"):
        dest_path = os.path.join(data_dir, filename)
        if os.path.exists(dest_path):
            continue
        response = requests.get(f"{UCI_BASE_URL}/{filename}", timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)


def create_spark_session():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("ProjetBigData")
        .config("spark.ui.port", "4040")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_train_data(spark, path="adult.data"):
    """Charge adult.data avec inférence de schéma (pour l'inspection initiale)."""
    df = spark.read.csv(
        path,
        header=False,
        inferSchema=True,
        sep=",",
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
    )
    return df.toDF(*COLUMNS)


def load_train_data_with_schema(spark, path="adult.data"):
    """Charge adult.data avec un schéma explicite (utilisé avant la génération de données)."""
    return spark.read.csv(path, header=False, schema=get_schema())


def load_test_data(spark, raw_path="adult.test", clean_path="adult_test_clean.csv"):
    """Nettoie les lignes de commentaires de adult.test puis charge le fichier résultant."""
    with open(raw_path) as f:
        cleaned_lines = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("|")
        ]

    with open(clean_path, "w") as f:
        f.write("\n".join(cleaned_lines))

    test_df = spark.read.csv(
        clean_path,
        header=False,
        inferSchema=True,
        sep=",",
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
    )
    return test_df.toDF(*COLUMNS)
