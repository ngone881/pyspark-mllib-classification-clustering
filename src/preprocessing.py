"""Nettoyage des données et génération de données synthétiques à grande échelle."""

import random

import numpy as np
from faker import Faker
from pyspark.sql.functions import col, when

from .data_loading import COLUMNS, get_schema

FILLNA_DEFAULTS = {
    "age": 30,
    "fnlwgt": 100000,
    "education_num": 9,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
}

CATEGORICAL_CHOICES = {
    "workclass": ["Private", "Self-emp-not-inc", "Federal-gov", "Local-gov", "State-gov"],
    "education": ["Bachelors", "HS-grad", "Some-college", "Masters", "Assoc-voc"],
    "marital_status": ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"],
    "occupation": ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales"],
    "relationship": ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Male", "Female"],
    "native_country": ["United-States", "Mexico", "Philippines", "Germany", "Canada"],
    "income": [">50K", "<=50K"],
}


def replace_question_marks(df, test_df):
    """Remplace les valeurs "?" par None dans les deux DataFrames."""
    for col_name in df.columns:
        df = df.withColumn(col_name, when(col(col_name) == "?", None).otherwise(col(col_name)))
        test_df = test_df.withColumn(col_name, when(col(col_name) == "?", None).otherwise(col(col_name)))
    return df, test_df


def fix_test_income_format(test_df):
    """Harmonise le format de la variable cible dans le fichier de test (">50K." -> ">50K")."""
    return test_df.withColumn(
        "income",
        when(col("income").contains("<=50K"), "<=50K").otherwise(">50K"),
    )


def drop_null_rows(df, test_df):
    return df.na.drop(), test_df.na.drop()


def prepare_base_data_for_generation(df):
    """Recharge/complète le DataFrame de base avant la génération (fillna + remplacement des "?")."""
    df = df.fillna(FILLNA_DEFAULTS)
    df = df.replace("?", None)
    return df


def _safe_clamp(value, min_val, max_val):
    try:
        val = int(round(float(value)))
        return min(max(val, min_val), max_val)
    except Exception:
        return min_val


def _get_cat_value(value, options):
    return str(value) if value and str(value) != "None" else random.choice(options)


def generate_big_data(spark, base_df, scale_factor=10, seed=2025):
    """Génère un dataset synthétique plus volumineux à partir des lignes réelles de base_df."""
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    base_data = base_df.collect()
    new_data = []

    for row in base_data:
        for _ in range(scale_factor):
            age = _safe_clamp(row.age * np.random.normal(1, 0.1), 17, 90)
            fnlwgt = _safe_clamp(row.fnlwgt * np.random.normal(1, 0.15), 10000, 2000000)
            education_num = _safe_clamp(row.education_num * np.random.normal(1, 0.1), 1, 16)
            capital_gain = _safe_clamp(row.capital_gain * np.random.normal(1, 0.5), 0, 99999)
            capital_loss = _safe_clamp(row.capital_loss * np.random.normal(1, 0.5), 0, 4356)
            hours_per_week = _safe_clamp(row.hours_per_week * np.random.normal(1, 0.1), 1, 99)

            workclass = _get_cat_value(row.workclass, CATEGORICAL_CHOICES["workclass"])
            education = _get_cat_value(row.education, CATEGORICAL_CHOICES["education"])
            marital_status = _get_cat_value(row.marital_status, CATEGORICAL_CHOICES["marital_status"])
            occupation = _get_cat_value(row.occupation, CATEGORICAL_CHOICES["occupation"])
            relationship = _get_cat_value(row.relationship, CATEGORICAL_CHOICES["relationship"])
            race = _get_cat_value(row.race, CATEGORICAL_CHOICES["race"])
            sex = _get_cat_value(row.sex, CATEGORICAL_CHOICES["sex"])
            native_country = _get_cat_value(row.native_country, CATEGORICAL_CHOICES["native_country"])
            income = _get_cat_value(row.income, CATEGORICAL_CHOICES["income"])

            new_data.append((
                int(age),
                str(workclass),
                int(fnlwgt),
                str(education),
                int(education_num),
                str(marital_status),
                str(occupation),
                str(relationship),
                str(race),
                str(sex),
                int(capital_gain),
                int(capital_loss),
                int(hours_per_week),
                str(native_country),
                str(income),
            ))

    return spark.createDataFrame(new_data, schema=get_schema())


def clean_big_data(big_df):
    """Supprime les lignes nulles du dataset généré et réapplique les noms de colonnes."""
    return big_df.na.drop().toDF(*COLUMNS)
