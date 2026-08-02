from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from config.settings import settings
from data.dataset_loader import load_dataset

GRID_KNN = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

def parse_args() -> argparse.Namespace:
    """Read paths while keeping the compact K-Means artifact as the default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.raw_data_dir / "dataset_saude_financeira_8000_empresas.csv",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=settings.models_dir / "kmeans_compact_v2.0.0.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.reports_dir / "kmeans" / "compact" / "validacao_knn",
    )
    return parser.parse_args()


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the original rows; feature engineering belongs to the K-Means pipeline."""
    dataset = load_dataset(path)
    required_splits = {"treino", "validacao", "teste"}
    if "split_ml" not in dataset or not required_splits.issubset(dataset["split_ml"]):
        raise ValueError(
            "Dataset must contain treino, validacao and teste in split_ml."
        )
    return dataset


def evaluate_predictions(expected: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Calculate agreement and per-cluster classification metrics.

    Here "expected" means the cluster predicted by the frozen K-Means artifact,
    not a human target. High agreement therefore supports local consistency of the
    segmentation; it is not evidence of causal or financial correctness by itself.
    """
    labels = sorted(np.unique(np.concatenate([expected, predicted])).tolist())
    precision, recall, f1_values, support = precision_recall_fscore_support(
        expected, predicted, labels=labels, zero_division=0
    )
    per_cluster = pd.DataFrame(
        {
            "cluster": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1_values,
            "support": support,
        }
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "classification_report": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels),
        "labels": labels,
        "per_cluster": per_cluster,
    }


def validate_clusters_with_knn(
    dataset: pd.DataFrame,
    kmeans_pipeline: Pipeline,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit KNN on train K-Means labels and compare it on validation and test.

    The K-Means pipeline was fitted earlier exclusively on train. Its transformer
    slice (all steps except K-Means) is reused verbatim, ensuring ratios, clipping,
    imputation and standardization are identical for both algorithms.
    """
    partitions = {
        name: dataset.loc[dataset["split_ml"] == name].copy()
        for name in ("treino", "validacao", "teste")
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("All split_ml partitions must contain at least one row.")

    feature_pipeline = kmeans_pipeline[:-1]
    transformed = {
        name: feature_pipeline.transform(frame) for name, frame in partitions.items()
    }
    kmeans_labels = {
        name: kmeans_pipeline.predict(frame) for name, frame in partitions.items()
    }

    distribution = pd.Series(kmeans_labels["treino"]).value_counts()
    folds = min(5, int(distribution.min()))
    if len(distribution) < 2 or folds < 2:
        raise ValueError(
            "Training partition needs at least two clusters and two rows per cluster."
        )

    # KNN receives already standardized numeric vectors. Adding another scaler here
    # would change the geometry used by K-Means and invalidate the comparison.
    knn_pipeline = Pipeline([("knn", KNeighborsClassifier())])
    search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=GRID_KNN,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
        n_jobs=-1,
        refit=True,
    )
    search.fit(transformed["treino"], kmeans_labels["treino"])

    predictions = {
        name: search.predict(transformed[name]) for name in ("validacao", "teste")
    }
    metrics = {
        name: evaluate_predictions(kmeans_labels[name], predictions[name])
        for name in ("validacao", "teste")
    }
    demonstrations = _select_demonstrations(
        partitions["teste"],
        transformed["teste"],
        kmeans_labels["teste"],
        predictions["teste"],
        kmeans_pipeline,
        search.best_estimator_,
    )
    interpretations = {
        "validation": _interpret_metrics(metrics["validacao"]),
        "test": _interpret_metrics(metrics["teste"]),
    }
    return {
        "model": search.best_estimator_,
        "best_parameters": search.best_params_,
        "cross_validation_macro_f1": float(search.best_score_),
        "training_distribution": distribution.sort_index().to_dict(),
        "validation": metrics["validacao"],
        "test": metrics["teste"],
        "interpretations": interpretations,
        "demonstrations": demonstrations,
    }


def _select_demonstrations(
    test: pd.DataFrame,
    transformed_test: np.ndarray,
    kmeans_labels: np.ndarray,
    knn_labels: np.ndarray,
    kmeans_pipeline: Pipeline,
    knn_pipeline: Pipeline,
) -> dict[str, Any]:
    """Choose typical cases and one boundary case from the untouched test partition."""
    centroid_distances = kmeans_pipeline.transform(test)
    assigned_distance = centroid_distances[np.arange(len(test)), kmeans_labels]
    knn_model: KNeighborsClassifier = knn_pipeline.named_steps["knn"]
    neighbor_distances, neighbor_indices = knn_model.kneighbors(transformed_test)
    training_labels = knn_model._y  # labels learned by the fitted KNN; no refit occurs.

    details = _neighbor_details(training_labels, neighbor_indices, neighbor_distances)
    frame = test[["id_empresa", "nome_empresa_ficticio"]].copy()
    frame["cluster_kmeans"] = kmeans_labels
    frame["cluster_knn"] = knn_labels
    frame["distancia_ao_centroide"] = assigned_distance
    frame["concordou"] = kmeans_labels == knn_labels
    frame["votos_maioria"] = [detail["majority_votes"] for detail in details]
    frame["margem_votos"] = [detail["vote_margin"] for detail in details]
    frame["vizinhos"] = details

    typical: list[dict[str, Any]] = []
    for cluster in sorted(np.unique(kmeans_labels)):
        candidates = frame.loc[
            (frame["cluster_kmeans"] == cluster) & frame["concordou"]
        ]
        if candidates.empty:
            candidates = frame.loc[frame["cluster_kmeans"] == cluster]
        typical.append(
            _record_to_dict(candidates.nsmallest(1, "distancia_ao_centroide").iloc[0])
        )

    # Prefer an actual disagreement with an even vote split; it represents a local
    # transition, not a failed prediction. If none disagree, choose the lowest margin.
    boundaries = frame.loc[~frame["concordou"]]
    if boundaries.empty:
        boundaries = frame
    boundary = boundaries.sort_values(
        ["margem_votos", "distancia_ao_centroide"], ascending=[True, False]
    ).iloc[0]
    return {"typical_companies": typical, "boundary_company": _record_to_dict(boundary)}


def _neighbor_details(
    training_labels: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, Any]]:
    """Summarize the neighbor votes that make each KNN decision explainable."""
    details: list[dict[str, Any]] = []
    for row_indices, row_distances in zip(indices, distances, strict=True):
        votes = pd.Series(training_labels[row_indices]).value_counts().sort_index()
        sorted_votes = votes.sort_values(ascending=False)
        second_count = int(sorted_votes.iloc[1]) if len(sorted_votes) > 1 else 0
        details.append(
            {
                "vote_distribution": {
                    str(label): int(count) for label, count in votes.items()
                },
                "majority_votes": int(sorted_votes.iloc[0]),
                "vote_margin": int(sorted_votes.iloc[0] - second_count),
                "mean_neighbor_distance": float(np.mean(row_distances)),
            }
        )
    return details


def _record_to_dict(record: pd.Series) -> dict[str, Any]:
    """Convert a selected company into JSON-safe, Streamlit-ready information."""
    values = record.to_dict()
    values["vizinhos"] = values["vizinhos"]
    return _json_safe(values)


def _interpret_metrics(metrics: dict[str, Any]) -> list[str]:
    """Create cautious, data-driven observations from recall and confusion counts."""
    messages: list[str] = []
    per_cluster: pd.DataFrame = metrics["per_cluster"]
    for row in per_cluster.itertuples(index=False):
        if row.recall >= 0.9:
            messages.append(
                f"Cluster {row.cluster} apresenta alta coesão local "
                f"(recall {row.recall:.1%})."
            )
    matrix = metrics["confusion_matrix"].copy()
    np.fill_diagonal(matrix, 0)
    if matrix.sum() > 0:
        real_index, predicted_index = np.unravel_index(matrix.argmax(), matrix.shape)
        labels = metrics["labels"]
        messages.append(
            f"A maior sobreposição ocorre entre Cluster {labels[real_index]} "
            f"e Cluster {labels[predicted_index]} "
            f"({int(matrix[real_index, predicted_index])} casos)."
        )
    if not messages:
        messages.append(
            "Não houve divergências relevantes entre K-Means e KNN nesta partição."
        )
    return messages


def export_validation(result: dict[str, Any], output_dir: Path) -> None:
    """Persist metrics, reports, confusion matrices and Streamlit demonstrations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_parameters": result["best_parameters"],
        "cross_validation_macro_f1": result["cross_validation_macro_f1"],
        "training_distribution": result["training_distribution"],
        "interpretations": result["interpretations"],
    }
    for partition in ("validation", "test"):
        metrics = result[partition]
        summary[partition] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
        metrics["per_cluster"].to_csv(
            output_dir / f"metricas_por_cluster_{partition}.csv", index=False
        )
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"Real C{label}" for label in metrics["labels"]],
            columns=[f"Previsto C{label}" for label in metrics["labels"]],
        ).to_csv(output_dir / f"matriz_confusao_{partition}.csv")
        (output_dir / f"classification_report_{partition}.json").write_text(
            json.dumps(
                _json_safe(metrics["classification_report"]),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    (output_dir / "resumo_validacao_knn.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "demonstracoes_knn.json").write_text(
        json.dumps(result["demonstrations"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy values to native JSON values and non-finite values to null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def print_result(result: dict[str, Any]) -> None:
    """Present a concise terminal summary for the validation and test partitions."""
    print("Best KNN parameters:", result["best_parameters"])
    print(f"Cross-validation macro F1: {result['cross_validation_macro_f1']:.4f}")
    for name in ("validation", "test"):
        metrics = result[name]
        print(f"\n{name.upper()}")
        print(f"Agreement (accuracy): {metrics['accuracy']:.2%}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(metrics["per_cluster"].to_string(index=False))
        print(
            pd.DataFrame(metrics["confusion_matrix"]).to_string(
                index=False, header=False
            )
        )
        for message in result["interpretations"][name]:
            print(f"- {message}")


def main() -> None:
    args = parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(
            f"K-Means artifact not found: {args.artifact}. "
            "Run run_kmeans_clustering.py first."
        )
    dataset = load_raw_dataset(args.input)
    kmeans_pipeline: Pipeline = joblib.load(args.artifact)
    result = validate_clusters_with_knn(dataset, kmeans_pipeline)
    export_validation(result, args.output_dir)
    print_result(result)


if __name__ == "__main__":
    main()