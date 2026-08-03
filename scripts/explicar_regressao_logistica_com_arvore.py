from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

from config.settings import settings
from data.dataset_loader import load_dataset


def add_frozen_cluster(
    dataset: pd.DataFrame,
    kmeans_model,
    cluster_feature: str,
) -> pd.DataFrame:
    """Adiciona o cluster produzido pelo K-Means já treinado."""
    enriched = dataset.copy()
    enriched[cluster_feature] = kmeans_model.predict(dataset).astype(str)
    return enriched


def fidelity_metrics(
    teacher_predictions: np.ndarray,
    tree_predictions: np.ndarray,
) -> dict[str, float]:
    """Mede quanto a árvore consegue imitar a regressão logística."""
    return {
        "mae": mean_absolute_error(
            teacher_predictions,
            tree_predictions,
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    teacher_predictions,
                    tree_predictions,
                )
            )
        ),
        "r2": r2_score(
            teacher_predictions,
            tree_predictions,
        ),
        "correlation": float(
            np.corrcoef(
                teacher_predictions,
                tree_predictions,
            )[0, 1]
        ),
    }


def main() -> None:
    dataset_path = (
        settings.raw_data_dir
        / "dataset_saude_financeira_8000_empresas.csv"
    )

    logistic_model_path = (
        settings.models_dir
        / "regressao_logistica_risco_v1.0.0.joblib"
    )

    metadata_path = (
        settings.models_dir
        / "regressao_logistica_risco_metadata.json"
    )

    output_dir = settings.reports_dir / "arvore_explicativa_logistica"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    numeric_features = metadata["numeric_features"]
    cluster_feature = metadata["cluster_feature"]
    feature_columns = numeric_features + [cluster_feature]

    kmeans_path = (
        settings.models_dir
        / metadata["kmeans_artifact"]
    )

    dataset = load_dataset(dataset_path)
    logistic_model = joblib.load(logistic_model_path)
    kmeans_model = joblib.load(kmeans_path)

    dataset = add_frozen_cluster(
        dataset,
        kmeans_model,
        cluster_feature,
    )

    splits = {
        name: dataset.loc[
            dataset["split_ml"] == name
        ].copy()
        for name in ("treino", "validacao", "teste")
    }

    # A regressão logística é o modelo professor.
    # A saída dela será o alvo aprendido pela árvore.
    teacher_probabilities = {
        name: logistic_model.predict_proba(
            frame[feature_columns]
        )[:, 1]
        for name, frame in splits.items()
    }

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cluster",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                [cluster_feature],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    candidates = []

    for max_depth in (3, 4):
        for min_samples_leaf in (25, 50, 100, 150):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "tree",
                        DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42,
                        ),
                    ),
                ]
            )

            pipeline.fit(
                splits["treino"][feature_columns],
                teacher_probabilities["treino"],
            )

            validation_prediction = pipeline.predict(
                splits["validacao"][feature_columns]
            )

            metrics = fidelity_metrics(
                teacher_probabilities["validacao"],
                validation_prediction,
            )

            candidates.append(
                {
                    "model": pipeline,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    **metrics,
                }
            )

    # Escolhe a árvore com maior fidelidade R² na validação.
    best = max(candidates, key=lambda item: item["r2"])
    best_model = best["model"]

    print("\nMelhores parâmetros:")
    print(f"max_depth: {best['max_depth']}")
    print(f"min_samples_leaf: {best['min_samples_leaf']}")

    results = []

    for split_name in ("treino", "validacao", "teste"):
        tree_prediction = best_model.predict(
            splits[split_name][feature_columns]
        )

        metrics = fidelity_metrics(
            teacher_probabilities[split_name],
            tree_prediction,
        )

        results.append(
            {
                "particao": split_name,
                **metrics,
            }
        )

        print(f"\n{split_name.upper()}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R² de fidelidade: {metrics['r2']:.4f}")
        print(f"Correlação: {metrics['correlation']:.4f}")

        prediction_report = splits[split_name][
            ["id_empresa"]
        ].copy()

        prediction_report["probabilidade_logistica"] = (
            teacher_probabilities[split_name]
        )

        prediction_report["probabilidade_arvore"] = (
            tree_prediction
        )

        prediction_report["erro_absoluto"] = np.abs(
            prediction_report["probabilidade_logistica"]
            - prediction_report["probabilidade_arvore"]
        )

        prediction_report.to_csv(
            output_dir
            / f"previsoes_{split_name}.csv",
            index=False,
        )

    pd.DataFrame(results).to_csv(
        output_dir / "metricas_fidelidade.csv",
        index=False,
    )

    fitted_preprocessor = best_model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()

    tree_rules = export_text(
        best_model.named_steps["tree"],
        feature_names=list(feature_names),
        decimals=3,
    )

    (output_dir / "regras_arvore.txt").write_text(
        tree_rules,
        encoding="utf-8",
    )

    feature_importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                best_model
                .named_steps["tree"]
                .feature_importances_
            ),
        }
    ).sort_values("importance", ascending=False)

    feature_importances.to_csv(
        output_dir / "importancia_variaveis.csv",
        index=False,
    )

    model_path = (
        settings.models_dir
        / "arvore_explicativa_logistica.joblib"
    )

    joblib.dump(best_model, model_path)

    print("\nRegras da árvore:")
    print(tree_rules)

    print(f"\nModelo salvo em: {model_path}")
    print(f"Relatórios salvos em: {output_dir}")


if __name__ == "__main__":
    main()