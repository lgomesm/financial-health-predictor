from __future__ import annotations

import json

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
from services.cashflow_regression_service import (
    CLUSTER_FEATURE,
    NUMERIC_FEATURES,
    CashflowFeatureTransformer,
)


def fidelity_metrics(
    teacher_predictions: np.ndarray,
    tree_predictions: np.ndarray,
) -> dict[str, float]:
    """Calcula o quanto a árvore consegue imitar a regressão linear."""
    correlation = np.corrcoef(
        teacher_predictions,
        tree_predictions,
    )[0, 1]

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
        "correlation": float(correlation),
    }


def main() -> None:
    dataset_path = (
        settings.raw_data_dir
        / "dataset_saude_financeira_8000_empresas.csv"
    )

    model_path = (
        settings.models_dir
        / "regressao_linear_fluxo_caixa_v1.0.0.joblib"
    )

    metadata_path = (
        settings.models_dir
        / "regressao_linear_fluxo_caixa_metadata.json"
    )

    output_dir = (
        settings.reports_dir
        / "arvore_explicativa_regressao_linear_fiel"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    raw_features = metadata["raw_features"]

    dataset = load_dataset(dataset_path)
    forecaster = joblib.load(model_path)

    splits = {
        name: dataset.loc[
            dataset["split_ml"] == name
        ].copy()
        for name in ("treino", "validacao", "teste")
    }

    # Modelo professor: regressão linear original.
    # A árvore aprenderá a margem futura prevista pela regressão.
    teacher_margin = {
        name: forecaster.predict_margin(frame)
        for name, frame in splits.items()
    }

    # Cria as mesmas variáveis internas utilizadas pela regressão linear.
    feature_transformer = CashflowFeatureTransformer()

    faithful_splits: dict[str, pd.DataFrame] = {}

    for name, frame in splits.items():
        enriched = frame.copy()

        # Usa exatamente o mesmo K-Means congelado do modelo original.
        enriched[CLUSTER_FEATURE] = (
            forecaster
            .kmeans_pipeline
            .predict(frame)
            .astype(str)
        )

        # Cria as razões financeiras:
        # margem_fcf_atual,
        # capital_giro_sobre_receita,
        # divida_sobre_receita.
        enriched = feature_transformer.transform(enriched)

        faithful_splits[name] = enriched

    faithful_features = list(NUMERIC_FEATURES) + [CLUSTER_FEATURE]

    # A árvore não precisa de StandardScaler.
    # Usamos imputação para valores ausentes e one-hot para o cluster.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(NUMERIC_FEATURES),
            ),
            (
                "cluster",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                [CLUSTER_FEATURE],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    candidates: list[dict[str, object]] = []

    # Árvores curtas para manter a interpretação simples.
    for max_depth in (3, 4):
        for min_samples_leaf in (25, 50, 100, 150):
            model = Pipeline(
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

            model.fit(
                faithful_splits["treino"][faithful_features],
                teacher_margin["treino"],
            )

            validation_prediction = model.predict(
                faithful_splits["validacao"][faithful_features]
            )

            metrics = fidelity_metrics(
                teacher_margin["validacao"],
                validation_prediction,
            )

            candidates.append(
                {
                    "model": model,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    **metrics,
                }
            )

    # Seleciona a maior fidelidade na validação.
    best = max(
        candidates,
        key=lambda item: float(item["r2"]),
    )

    best_model = best["model"]

    print("\nMelhores parâmetros:")
    print(f"max_depth: {best['max_depth']}")
    print(
        "min_samples_leaf:",
        best["min_samples_leaf"],
    )

    results: list[dict[str, float | str]] = []

    for split_name in ("treino", "validacao", "teste"):
        tree_margin = best_model.predict(
            faithful_splits[split_name][faithful_features]
        )

        metrics = fidelity_metrics(
            teacher_margin[split_name],
            tree_margin,
        )

        results.append(
            {
                "particao": split_name,
                **metrics,
            }
        )

        print(f"\n{split_name.upper()}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(
            f"R² de fidelidade: {metrics['r2']:.4f}"
        )
        print(
            f"Correlação: {metrics['correlation']:.4f}"
        )

        revenue = pd.to_numeric(
            splits[split_name]["receita_liquida_usd_m"],
            errors="coerce",
        ).to_numpy(dtype=float)

        teacher_fcf = (
            teacher_margin[split_name] * revenue
        )

        tree_fcf = tree_margin * revenue

        report_columns = [
            column
            for column in (
                "id_empresa",
                "nome_empresa_ficticio",
            )
            if column in splits[split_name].columns
        ]

        report = splits[split_name][
            report_columns
        ].copy()

        report["margem_prevista_regressao"] = (
            teacher_margin[split_name]
        )

        report["margem_prevista_arvore"] = (
            tree_margin
        )

        report["fcf_previsto_regressao_usd_m"] = (
            teacher_fcf
        )

        report["fcf_previsto_arvore_usd_m"] = (
            tree_fcf
        )

        report["erro_absoluto_margem"] = np.abs(
            teacher_margin[split_name]
            - tree_margin
        )

        report["erro_absoluto_fcf_usd_m"] = np.abs(
            teacher_fcf
            - tree_fcf
        )

        report.to_csv(
            output_dir
            / f"previsoes_{split_name}.csv",
            index=False,
        )

    pd.DataFrame(results).to_csv(
        output_dir / "metricas_fidelidade.csv",
        index=False,
    )

    fitted_preprocessor = (
        best_model.named_steps["preprocessor"]
    )

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

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
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importances.to_csv(
        output_dir / "importancia_variaveis.csv",
        index=False,
    )

    saved_model_path = (
        settings.models_dir
        / "arvore_explicativa_regressao_linear_fiel.joblib"
    )

    joblib.dump(
        best_model,
        saved_model_path,
    )

    summary = {
        "teacher_model": metadata["artifact_file"],
        "target_explained": (
            "margem futura prevista pela regressão linear"
        ),
        "raw_features": raw_features,
        "faithful_features": faithful_features,
        "best_max_depth": best["max_depth"],
        "best_min_samples_leaf": (
            best["min_samples_leaf"]
        ),
        "validation_fidelity": {
            "mae": best["mae"],
            "rmse": best["rmse"],
            "r2": best["r2"],
            "correlation": best["correlation"],
        },
    }

    (output_dir / "resumo.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRegras da árvore:")
    print(tree_rules)

    print(
        f"\nModelo salvo em: {saved_model_path}"
    )

    print(
        f"Relatórios salvos em: {output_dir}"
    )


if __name__ == "__main__":
    main()