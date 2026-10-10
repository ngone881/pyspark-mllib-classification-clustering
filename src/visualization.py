"""Fonctions de visualisation : sauvegarde des figures dans reports/figures/."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _savefig(filename, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, model_name, filename, output_dir="reports/figures"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    return _savefig(filename, output_dir)


def plot_clusters(pred_df, numeric_cols, scatter_x, scatter_y, box_col, title, filename,
                   output_dir="reports/figures", sample_fraction=0.1, seed=42, limit=300000,
                   features_col="scaled_numeric_features"):
    """Visualise un clustering : scatter (scatter_x vs scatter_y) + boxplot (box_col par cluster).

    Fonction unique réutilisable remplaçant les multiples versions dupliquées de plot_clusters
    du script original (une par paire de variables à visualiser).
    """
    sample = pred_df.sample(False, sample_fraction, seed).limit(limit).toPandas()
    features = sample[features_col].apply(lambda v: v.toArray())
    features = pd.DataFrame(features.tolist(), columns=numeric_cols)
    features["cluster"] = sample["prediction"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=features, x=scatter_x, y=scatter_y, hue="cluster", palette="viridis")
    plt.title(f"{title} - {scatter_x} vs {scatter_y}")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=features, x="cluster", y=box_col, palette="Set2")
    plt.title(f"{title} - {box_col} par cluster")

    return _savefig(filename, output_dir)


def plot_cluster_distribution_pie(cluster_counts, labels, title, filename, output_dir="reports/figures"):
    plt.figure(figsize=(5, 5))
    cluster_counts.plot.pie(autopct="%1.1f%%", labels=labels)
    plt.title(title)
    plt.ylabel("")
    return _savefig(filename, output_dir)


def plot_classification_metrics(results_df, filename, output_dir="reports/figures"):
    """Barplots Accuracy/Precision/Recall/F1 Score pour chaque modèle de classification."""
    metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1 Score"] if m in results_df.columns]

    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, len(metrics), i)
        sns.barplot(data=results_df, x="Modèle", y=metric, palette="Set2")
        plt.title(metric)
        plt.ylim(80, 90)
        plt.xticks(rotation=30)

    plt.suptitle("Performance des modèles de classification supervisée")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _savefig(filename, output_dir)


def plot_clustering_silhouette(silhouette_scores, filename, output_dir="reports/figures"):
    """Barplot comparant les scores de silhouette des algorithmes de clustering."""
    df = pd.DataFrame(
        [{"Modèle": name, "Silhouette Score": score} for name, score in silhouette_scores.items()]
    )
    plt.figure(figsize=(6, 5))
    sns.barplot(x="Modèle", y="Silhouette Score", data=df, palette="Set1")
    plt.title("Score Silhouette clustering")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    return _savefig(filename, output_dir)
