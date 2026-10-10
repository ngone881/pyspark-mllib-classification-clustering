"""Pipeline complet : chargement -> nettoyage -> génération -> feature engineering ->
classification -> clustering -> sauvegarde des résultats et des figures."""

import pandas as pd
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.types import NumericType, StringType

from src import classification, clustering, data_loading, feature_engineering, preprocessing, visualization

DATA_DIR = "."
FIGURES_DIR = "reports/figures"


def main():
    data_loading.download_dataset(DATA_DIR)
    spark = data_loading.create_spark_session()
    print("Spark version:", spark.version)

    # 1. Chargement et inspection du dataset réel
    df = data_loading.load_train_data(spark, f"{DATA_DIR}/adult.data")
    df.printSchema()
    df.show(5)

    total_rows = df.count()
    print(f"Nombre total de lignes : {total_rows}")

    numeric_cols_raw = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    categorical_cols_raw = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    print(f"Colonnes numériques : {numeric_cols_raw}")
    print(f"Colonnes catégorielles : {categorical_cols_raw}")

    test_df = data_loading.load_test_data(spark, f"{DATA_DIR}/adult.test", f"{DATA_DIR}/adult_test_clean.csv")

    df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()
    df.describe().show()

    # 2. Nettoyage des données
    df, test_df = preprocessing.replace_question_marks(df, test_df)
    test_df = preprocessing.fix_test_income_format(test_df)

    df_clean, test_clean = preprocessing.drop_null_rows(df, test_df)
    print(f"Train: {df_clean.count()} instances")
    print(f"Test: {test_clean.count()} instances")

    # 3. Génération de données synthétiques à grande échelle
    df_schema = data_loading.load_train_data_with_schema(spark, f"{DATA_DIR}/adult.data")
    df_schema = preprocessing.prepare_base_data_for_generation(df_schema)

    print("Début de la génération des données...")
    big_df = preprocessing.generate_big_data(spark, df_schema, scale_factor=10, seed=2025)
    print(f"Nombre total de lignes générées: {big_df.count():,}")
    big_df.write.mode("overwrite").parquet(f"{DATA_DIR}/big_adult_data.parquet")
    print("Données sauvegardées avec succès dans 'big_adult_data.parquet'")

    big_df = spark.read.parquet(f"{DATA_DIR}/big_adult_data.parquet")
    df_clean_big = preprocessing.clean_big_data(big_df)

    # 4. Feature engineering (classification)
    classification_pipeline = feature_engineering.build_classification_pipeline()
    train_data, test_data = feature_engineering.split_train_test(df_clean_big, train_ratio=0.8, seed=2025)
    _, train_transformed, test_transformed = feature_engineering.fit_transform_pipeline(
        classification_pipeline, train_data, test_data
    )
    print(f"Nombre d'échantillons d'entraînement : {train_transformed.count():,}")
    print(f"Nombre d'échantillons de test : {test_transformed.count():,}")

    train_transformed.write.mode("overwrite").parquet(f"{DATA_DIR}/train_data_transformed.parquet")
    test_transformed.write.mode("overwrite").parquet(f"{DATA_DIR}/test_data_transformed.parquet")

    train_data, test_data = feature_engineering.scale_features(train_transformed, test_transformed)

    # 5. Classification supervisée
    results_df = classification.run_classification_models(train_data, test_data, FIGURES_DIR)
    print("\nRésumé des scores :")
    print(results_df)

    # 6. Clustering non supervisé
    numeric_cols = feature_engineering.NUMERIC_COLS
    clustering_pipeline = feature_engineering.build_clustering_pipeline(numeric_cols)
    cluster_data = clustering_pipeline.fit(df_clean_big).transform(df_clean_big)

    kmeans_model, kmeans_pred = clustering.run_kmeans(cluster_data, k=2, seed=2025)
    bkm_model, bkm_pred = clustering.run_bisecting_kmeans(cluster_data, k=2, seed=2025)

    print("======================================")
    print("ANALYSE COMPLÈTE DES CLUSTERS (k=2)")
    print("======================================")
    clustering.analyze_clusters_safe(kmeans_pred, kmeans_model, "K-Means", numeric_cols)
    clustering.analyze_clusters_safe(bkm_pred, bkm_model, "Bisecting K-Means", numeric_cols)

    visualization.plot_clusters(
        kmeans_pred, numeric_cols, "age", "hours_per_week", "education_num",
        "K-Means - 2 clusters", "clusters_kmeans_age_hours.png", FIGURES_DIR,
    )
    visualization.plot_clusters(
        bkm_pred, numeric_cols, "age", "hours_per_week", "education_num",
        "Bisecting K-Means - 2 clusters", "clusters_bkm_age_hours.png", FIGURES_DIR,
    )
    visualization.plot_clusters(
        kmeans_pred, numeric_cols, "education_num", "capital_gain", "education_num",
        "K-Means - 2 clusters", "clusters_kmeans_edu_capgain.png", FIGURES_DIR,
    )
    visualization.plot_clusters(
        bkm_pred, numeric_cols, "education_num", "capital_gain", "education_num",
        "Bisecting K-Means - 2 clusters", "clusters_bkm_edu_capgain.png", FIGURES_DIR,
    )

    print("\nTable de correspondance entre les clusters:")
    clustering.compare_cluster_assignments(cluster_data, kmeans_model, bkm_model).show()

    kmeans_summary = clustering.cluster_summary(kmeans_pred, numeric_cols)
    bkm_summary = clustering.cluster_summary(bkm_pred, numeric_cols)
    comparison_df = clustering.compare_cluster_summaries(kmeans_summary, bkm_summary)
    print("\nComparaison des moyennes par cluster entre K-Means et Bisecting K-Means :")
    print(comparison_df)

    counts = (
        kmeans_pred.groupBy("prediction").count().orderBy("prediction").toPandas().set_index("prediction")["count"]
    )
    visualization.plot_cluster_distribution_pie(
        counts,
        labels=["Cluster 0 = Revenue >= 50k$", "Cluster 1 = Revenue < 50k$"],
        title="Répartition des clusters",
        filename="cluster_distribution_pie_kmeans.png",
        output_dir=FIGURES_DIR,
    )

    silhouette_kmeans = clustering.compute_silhouette(kmeans_pred)
    silhouette_bkm = clustering.compute_silhouette(bkm_pred)
    print(f"Silhouette Score - K-Means: {silhouette_kmeans:.3f}")
    print(f"Silhouette Score - Bisecting K-Means: {silhouette_bkm:.3f}")

    # 7. Figures de synthèse
    visualization.plot_classification_metrics(
        pd.DataFrame(classification.CLASSIFICATION_SUMMARY),
        "classification_metrics_summary.png",
        FIGURES_DIR,
    )
    visualization.plot_clustering_silhouette(
        {"K-Means": silhouette_kmeans, "Bisecting K-Means": silhouette_bkm},
        "clustering_silhouette_comparison.png",
        FIGURES_DIR,
    )

    spark.stop()


if __name__ == "__main__":
    main()
