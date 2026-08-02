from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge

from config.settings import settings
from data.dataset_loader import load_dataset
from services.cashflow_regression_service import (
    CLUSTER_FEATURE,
    CashflowForecaster,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    TARGET,
    TARGET_MARGIN,
    add_target_margin,
    build_cashflow_pipeline,
    coefficients_table,
    evaluate_forecast,
    metrics_by_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.raw_data_dir / "dataset_saude_financeira_8000_empresas.csv",
    )
    parser.add_argument(
        "--kmeans-artifact",
        type=Path,
        default=settings.models_dir / "kmeans_compact_v2.0.0.joblib",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=settings.reports_dir / "fluxo_caixa_linear"
    )
    return parser.parse_args()


def add_frozen_cluster(dataset: pd.DataFrame, kmeans_pipeline: Any) -> pd.DataFrame:
    """Attach K-Means context from a pipeline fitted only on its original train set."""
    enriched = dataset.copy()
    enriched[CLUSTER_FEATURE] = kmeans_pipeline.predict(dataset).astype(str)
    return enriched


def add_display_risk(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add frozen logistic-risk labels for display only; they are not regression inputs."""
    model_path = settings.models_dir / "regressao_logistica_risco_v1.0.0.joblib"
    metadata_path = settings.models_dir / "regressao_logistica_risco_metadata.json"
    enriched = dataset.copy()
    enriched["nivel_risco"] = "Indisponível"
    if not (model_path.is_file() and metadata_path.is_file()):
        return enriched
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["numeric_features"] + [metadata["cluster_feature"]]
    probability = joblib.load(model_path).predict_proba(enriched[features])[:, 1]
    enriched["nivel_risco"] = np.select(
        [probability >= metadata["high_threshold"], probability >= metadata["moderate_threshold"]],
        ["Alto", "Moderado"],
        default="Baixo",
    )
    enriched["probabilidade_deterioracao"] = probability
    return enriched


def calculate_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF on train-derived ratios to flag unstable conditional coefficients."""
    prepared = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(features), columns=features.columns
    )
    values = []
    for feature in prepared:
        others = prepared.drop(columns=feature)
        r_squared = LinearRegression().fit(others, prepared[feature]).score(
            others, prepared[feature]
        )
        values.append(
            {"feature": feature, "vif": float("inf") if r_squared >= 0.999999 else 1 / (1 - r_squared)}
        )
    return pd.DataFrame(values).sort_values("vif", ascending=False)


def partition_diagnostics(frame: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float | int]:
    """Describe target dispersion so partition-level RMSE can be interpreted fairly."""
    target = frame[TARGET]
    return {
        "empresas": int(len(frame)),
        "media_target": float(target.mean()),
        "mediana_target": float(target.median()),
        "desvio_padrao_target": float(target.std()),
        "min_target": float(target.min()),
        "p01_target": float(target.quantile(0.01)),
        "p25_target": float(target.quantile(0.25)),
        "p75_target": float(target.quantile(0.75)),
        "p95_target": float(target.quantile(0.95)),
        "p99_target": float(target.quantile(0.99)),
        "max_target": float(target.max()),
        "empresas_grandes": int(frame["porte_empresa"].eq("Grande").sum()),
        "maior_erro_absoluto": float(prediction["erro_absoluto_usd_m"].max()),
    }


def select_ridge_for_robustness(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> tuple[float, Any]:
    """Choose Ridge alpha on validation only; this does not replace the OLS baseline."""
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    best_alpha, best_model, best_rmse = candidates[0], None, float("inf")
    for alpha in candidates:
        candidate = build_cashflow_pipeline(Ridge(alpha=alpha))
        candidate.fit(train[features], train[TARGET_MARGIN])
        prediction = candidate.predict(validation[features]) * validation["receita_liquida_usd_m"].to_numpy()
        rmse = evaluate_forecast(validation[TARGET].to_numpy(), prediction)["rmse"]
        if rmse < best_rmse:
            best_alpha, best_model, best_rmse = alpha, candidate, rmse
    return best_alpha, best_model


def fit_and_evaluate(dataset: pd.DataFrame) -> dict[str, Any]:
    """Fit OLS on train, monitor validation, and keep the test set for final evaluation."""
    required = set(RAW_COLUMNS + (TARGET, "split_ml", "id_empresa", "nome_empresa_ficticio"))
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError("Colunas ausentes: " + ", ".join(missing))
    prepared = add_target_margin(dataset)
    partitions = {
        name: prepared.loc[prepared["split_ml"].eq(name)].reset_index(drop=True)
        for name in ("treino", "validacao", "teste")
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("São necessários registros de treino, validação e teste com alvo válido.")
    features = list(RAW_COLUMNS) + [CLUSTER_FEATURE]
    model = build_cashflow_pipeline()
    train = partitions["treino"]
    model.fit(train[features], train[TARGET_MARGIN])

    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, float | None]] = {}
    margin_metrics: dict[str, dict[str, float | None]] = {}
    for name, frame in partitions.items():
        predicted_margin = model.predict(frame[features])
        actual = frame[TARGET].to_numpy(dtype=float)
        predicted = predicted_margin * frame["receita_liquida_usd_m"].to_numpy(dtype=float)
        residual = actual - predicted
        predictions[name] = frame.assign(
            margem_fcf_real=frame[TARGET_MARGIN],
            margem_fcf_prevista=predicted_margin,
            fluxo_caixa_livre_real_usd_m=actual,
            fluxo_caixa_livre_previsto_usd_m=predicted,
            residuo_usd_m=residual,
            erro_absoluto_usd_m=np.abs(residual),
        )
        metrics[name] = evaluate_forecast(actual, predicted)
        margin_metrics[name] = evaluate_forecast(
            frame[TARGET_MARGIN].to_numpy(dtype=float), predicted_margin
        )

    transformed_train = model.named_steps["ratios"].transform(train[features])
    test_prediction = predictions["teste"]
    groups = {
        "porte": metrics_by_group(
            test_prediction, test_prediction["fluxo_caixa_livre_real_usd_m"].to_numpy(),
            test_prediction["fluxo_caixa_livre_previsto_usd_m"].to_numpy(), "porte_empresa"
        ),
        "setor": metrics_by_group(
            test_prediction, test_prediction["fluxo_caixa_livre_real_usd_m"].to_numpy(),
            test_prediction["fluxo_caixa_livre_previsto_usd_m"].to_numpy(), "setor_economico"
        ),
    }
    ridge_alpha, ridge_model = select_ridge_for_robustness(
        train, partitions["validacao"], features
    )
    test = partitions["teste"]
    ridge_prediction = ridge_model.predict(test[features]) * test["receita_liquida_usd_m"].to_numpy()
    baseline_predictions = {
        "Persistência: FCF atual": test["fluxo_caixa_livre_usd_m"].to_numpy(dtype=float),
        "Margem média do treino": train[TARGET_MARGIN].mean()
        * test["receita_liquida_usd_m"].to_numpy(dtype=float),
        "Regressão Linear OLS": predictions["teste"]["fluxo_caixa_livre_previsto_usd_m"].to_numpy(),
        f"Ridge (alpha={ridge_alpha})": ridge_prediction,
    }
    baseline_comparison = pd.DataFrame(
        [{"modelo": name, **evaluate_forecast(test[TARGET].to_numpy(), values)} for name, values in baseline_predictions.items()]
    )
    return {
        "model": model,
        "partitions": partitions,
        "predictions": predictions,
        "metrics": metrics,
        "margin_metrics": margin_metrics,
        "groups": groups,
        "coefficients": coefficients_table(model),
        "intercept": float(model.named_steps["regression"].intercept_),
        "correlation": transformed_train.loc[:, NUMERIC_FEATURES].corr(),
        "vif": calculate_vif(transformed_train.loc[:, NUMERIC_FEATURES]),
        "baseline_comparison": baseline_comparison,
        "ridge_alpha": ridge_alpha,
        "partition_diagnostics": {
            name: partition_diagnostics(partitions[name], predictions[name])
            for name in partitions
        },
    }


def save_plots(result: dict[str, Any], output_dir: Path) -> None:
    """Save diagnostics based exclusively on held-out test residuals."""
    test = result["predictions"]["teste"]
    actual = test["fluxo_caixa_livre_real_usd_m"]
    predicted = test["fluxo_caixa_livre_previsto_usd_m"]
    residual = test["residuo_usd_m"]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual, predicted, alpha=0.45)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axis.plot(limits, limits, "--", color="gray")
    axis.set(title="FCF previsto versus real", xlabel="FCF real (USD mi)", ylabel="FCF previsto (USD mi)")
    figure.tight_layout(); figure.savefig(output_dir / "previsto_vs_real.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(residual, bins=30)
    axis.set(title="Distribuição dos resíduos", xlabel="Real − previsto (USD mi)", ylabel="Empresas")
    figure.tight_layout(); figure.savefig(output_dir / "histograma_residuos.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predicted, residual, alpha=0.45); axis.axhline(0, linestyle="--", color="gray")
    axis.set(title="Resíduos versus FCF previsto", xlabel="FCF previsto (USD mi)", ylabel="Resíduo (USD mi)")
    figure.tight_layout(); figure.savefig(output_dir / "residuos_vs_previsto.png", dpi=150); plt.close(figure)
    coefficients = result["coefficients"].sort_values("coefficient")
    figure, axis = plt.subplots(figsize=(9, 6)); axis.barh(coefficients["feature"], coefficients["coefficient"])
    axis.set(title="Coeficientes OLS padronizados", xlabel="Coeficiente")
    figure.tight_layout(); figure.savefig(output_dir / "coeficientes.png", dpi=150); plt.close(figure)


def export_results(
    result: dict[str, Any], output_dir: Path, input_path: Path, kmeans_path: Path, kmeans_pipeline: Any
) -> None:
    """Persist model, reports, diagnostics, and the lineage needed for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    for name, prediction in result["predictions"].items():
        prediction.to_csv(output_dir / f"previsoes_{name}.csv", index=False)
    result["coefficients"].to_csv(output_dir / "coeficientes.csv", index=False)
    result["vif"].to_csv(output_dir / "vif.csv", index=False)
    result["correlation"].to_csv(output_dir / "matriz_correlacao.csv")
    result["baseline_comparison"].to_csv(output_dir / "comparacao_baselines.csv", index=False)
    pd.DataFrame(result["partition_diagnostics"]).T.to_csv(
        output_dir / "diagnostico_particoes.csv"
    )
    for name, group_metrics in result["groups"].items():
        group_metrics.to_csv(output_dir / f"metricas_por_{name}.csv", index=False)
    model_path = settings.models_dir / "regressao_linear_fluxo_caixa_v1.0.0.joblib"
    # Persisting this composite object prevents callers from forgetting either the
    # frozen K-Means cluster step or the margin-to-dollar reconstruction step.
    forecaster = CashflowForecaster(kmeans_pipeline, result["model"])
    joblib.dump(forecaster, model_path)
    cluster_encoder = result["model"].named_steps["preprocessor"].named_transformers_["cluster"]
    cluster_reference = str(cluster_encoder.categories_[0][0])
    interpretations = [
        "A previsão é uma margem futura de FCF reconstruída pela receita atual; portanto, pressupõe manutenção dessa escala de receita.",
        "Coeficientes são associações condicionais após padronização, não efeitos causais.",
    ]
    for row in result["coefficients"].head(3).itertuples(index=False):
        if row.feature.startswith(f"{CLUSTER_FEATURE}_"):
            cluster = row.feature.removeprefix(f"{CLUSTER_FEATURE}_")
            interpretations.append(
                f"Pertencer ao Cluster {cluster} está associado a uma margem futura de FCF "
                f"{abs(row.coefficient) * 100:.2f} pontos percentuais "
                f"{'maior' if row.coefficient >= 0 else 'menor'} que o Cluster {cluster_reference}, "
                "mantendo as demais variáveis constantes."
            )
            continue
        direction = "maior" if row.direction == "aumenta" else "menor"
        interpretations.append(f"{row.feature}: valor {direction} está associado a maior margem futura de FCF, mantendo as demais variáveis constantes.")
    summary = {
        "metrics_fcf_reconstructed": result["metrics"],
        "metrics_margin": result["margin_metrics"],
        "intercept": result["intercept"],
        "model": "Ordinary Least Squares (LinearRegression)",
        "ridge_robustness_alpha": result["ridge_alpha"],
        "hyperparameters": "OLS não possui hiperparâmetros; Ridge é apenas uma análise de robustez selecionada na validação.",
        "interpretation": interpretations,
    }
    (output_dir / "resumo_fluxo_caixa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"artifact_version": "1.0.0", "target": TARGET, "target_transformation": "target / receita_liquida_usd_m", "artifact_file": model_path.name, "dataset_file": input_path.name, "dataset_hash": file_hash(input_path), "kmeans_artifact": kmeans_path.name, "kmeans_artifact_hash": file_hash(kmeans_path), "cluster_reference": cluster_reference, "raw_features": list(RAW_COLUMNS), "trained_at_utc": datetime.now(UTC).isoformat()}
    (settings.models_dir / "regressao_linear_fluxo_caixa_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(result, output_dir)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.kmeans_artifact.is_file():
        raise FileNotFoundError("Artefato K-Means não encontrado. Execute o K-Means compacto primeiro.")
    kmeans_pipeline = joblib.load(args.kmeans_artifact)
    dataset = add_display_risk(add_frozen_cluster(load_dataset(args.input), kmeans_pipeline))
    result = fit_and_evaluate(dataset)
    export_results(result, args.output_dir, args.input, args.kmeans_artifact, kmeans_pipeline)
    for name, metrics in result["metrics"].items():
        print(f"{name.upper()}: MAE {metrics['mae']:.3f}; RMSE {metrics['rmse']:.3f}; R² {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
