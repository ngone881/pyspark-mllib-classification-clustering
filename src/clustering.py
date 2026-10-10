"""Clustering non supervisé (K-Means, Bisecting K-Means) et analyse des clusters.

Le script original contenait deux implémentations quasi identiques du clustering
(une avec seed=2025 sur "scaled_numeric_features", une avec seed=42 sur "scaled_features").
Seule la première est conservée ici : c'est elle qui alimente toutes les analyses et
visualisations en aval (silhouette, comparaison des clusters, etc.).
"""

import pandas as pd
from pyspark.ml.clustering import BisectingKMeans, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, mean, stddev


def run_kmeans(data, k=2, seed=2025, features_col="scaled_numeric_features"):
    kmeans = KMeans(featuresCol=features_col, k=k, seed=seed)
    model = kmeans.fit(data)
    return model, model.transform(data)


def run_bisecting_kmeans(data, k=2, seed=2025, features_col="scaled_numeric_features"):
    bkm = BisectingKMeans(featuresCol=features_col, k=k, seed=seed)
    model = bkm.fit(data)
    return model, model.transform(data)


def analyze_clusters_safe(results_df, model, algorithm_name, feature_cols):
    print(f"\n=== Analyse des clusters - {algorithm_name} ===")

    print("\nNombre de points par cluster:")
    results_df.groupBy("prediction").count().orderBy("prediction").show()

    centers = model.clusterCenters()
    print("\nCentroïdes des clusters (features standardisées):")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {[round(x, 2) for x in center]}")

    print("\nCaractéristiques moyennes par cluster (valeurs originales):")
    for col_name in feature_cols:
        print(f"\nVariable: {col_name}")
        results_df.groupBy("prediction").agg(
            mean(col(col_name)).alias("moyenne"),
            stddev(col(col_name)).alias("ecart_type"),
        ).orderBy("prediction").show()


def compute_silhouette(pred_df, features_col="scaled_numeric_features"):
    evaluator = ClusteringEvaluator(
        predictionCol="prediction",
        featuresCol=features_col,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    return evaluator.evaluate(pred_df)


def compare_cluster_assignments(cluster_data, kmeans_model, bkm_model):
    """Table de correspondance entre les clusters K-Means et Bisecting K-Means.

    Les deux modèles sont appliqués au même DataFrame (même ordre de lignes), avec des noms
    de colonnes de prédiction distincts, pour comparer les assignations sans cross join.
    """
    kmeans_renamed = kmeans_model.transform(cluster_data, params={kmeans_model.predictionCol: "kmeans_cluster"})
    combined = bkm_model.transform(kmeans_renamed, params={bkm_model.predictionCol: "bkm_cluster"})
    return combined.groupBy("kmeans_cluster", "bkm_cluster").count().orderBy("kmeans_cluster", "bkm_cluster")


def cluster_summary(pred_df, numeric_cols, features_col="scaled_numeric_features"):
    """Moyenne des variables numériques et taille de chaque cluster (features standardisées)."""
    data = pred_df.select(features_col, "prediction").toPandas()
    features = data[features_col].apply(lambda x: x.toArray())
    features_df = pd.DataFrame(features.tolist(), columns=numeric_cols)
    features_df["cluster"] = data["prediction"]

    summary = features_df.groupby("cluster").mean().round(3)
    summary["count"] = features_df["cluster"].value_counts().sort_index().values
    return summary


def compare_cluster_summaries(kmeans_summary, bkm_summary):
    """Fusionne les résumés K-Means et Bisecting K-Means côte à côte."""
    return pd.concat(
        [kmeans_summary.add_suffix(" (KMeans)"), bkm_summary.add_suffix(" (Bisecting KMeans)")],
        axis=1,
    )
