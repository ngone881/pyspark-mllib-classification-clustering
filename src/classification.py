"""Entraînement et évaluation des modèles de classification supervisée."""

import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report

from .visualization import plot_confusion_matrix

# Métriques détaillées (Accuracy/Precision/Recall/F1) obtenues lors de l'exécution de référence
# du projet, reportées ici telles quelles pour le graphique de synthèse (le MulticlassClassificationEvaluator
# de train_evaluate ne calcule que accuracy et f1).
CLASSIFICATION_SUMMARY = [
    {"Modèle": "Régression Logistique", "Accuracy": 84.76, "Precision": 84.14, "Recall": 84.76, "F1 Score": 84.25},
    {"Modèle": "Arbre de Décision", "Accuracy": 83.64, "Precision": 82.82, "Recall": 83.64, "F1 Score": 82.35},
    {"Modèle": "SVM", "Accuracy": 84.66, "Precision": 83.96, "Recall": 84.66, "F1 Score": 84.01},
]


def train_evaluate(model, name, train_data, test_data, output_dir="reports/figures"):
    trained_model = model.fit(train_data)
    predictions = trained_model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    y_true = predictions.select("label").toPandas()
    y_pred = predictions.select("prediction").toPandas()

    filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(y_true, y_pred, name, filename, output_dir)

    print(f"\nRapport de classification - {name}")
    print(classification_report(y_true, y_pred, digits=3))

    return {
        "Modèle": name,
        "Accuracy": accuracy,
        "F1 Score": f1,
    }


def run_classification_models(train_data, test_data, output_dir="reports/figures"):
    from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, LogisticRegression

    models = [
        (LogisticRegression(featuresCol="scaledFeatures", labelCol="label"), "Régression Logistique"),
        (DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol="label"), "Arbre de Décision"),
        (LinearSVC(featuresCol="scaledFeatures", labelCol="label"), "SVM"),
    ]

    results = [
        train_evaluate(model, name, train_data, test_data, output_dir)
        for model, name in models
    ]

    return pd.DataFrame(results)
