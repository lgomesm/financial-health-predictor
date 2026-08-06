"""Gerar o dataset analítico consolidado com as saídas de todos os modelos."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from data.dataset_loader import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.raw_data_dir / "dataset_saude_financeira_8000_empresas.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.processed_data_dir / "empresas_com_resultados_modelos.csv",
    )
    return parser.parse_args()


def risk_levels(probabilities: np.ndarray, moderate: float, high: float) -> np.ndarray:
    """Apply the persisted operational thresholds from the logistic model."""
    return np.select(
        [probabilities >= high, probabilities >= moderate],
        ["Alto", "Moderado"],
        default="Baixo",
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consolidation_status(split: pd.Series) -> np.ndarray:
    """Distinguish in-sample rows from the holdout rows used for evaluation."""
    return np.select(
        [split.eq("treino"), split.isin(["validacao", "teste"])],
        ["treino_in_sample", "holdout"],
        default="nao_classificado",
    )


def main() -> None:
    args = parse_args()
    kmeans_path = settings.models_dir / "kmeans_compact_v2.0.0.joblib"
    logistic_path = settings.models_dir / "regressao_logistica_risco_v1.0.0.joblib"
    logistic_metadata_path = settings.models_dir / "regressao_logistica_risco_metadata.json"
    linear_path = settings.models_dir / "regressao_linear_fluxo_caixa_v1.0.0.joblib"
    required = (kmeans_path, logistic_path, logistic_metadata_path, linear_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Artefatos ausentes: " + ", ".join(missing))

    dataset = load_dataset(args.input)
    kmeans = joblib.load(kmeans_path)
    logistic = joblib.load(logistic_path)
    forecaster = joblib.load(linear_path)
    logistic_metadata = json.loads(logistic_metadata_path.read_text(encoding="utf-8"))

    consolidated = dataset.copy()
    # All predictions are generated from frozen artifacts. No model is fitted here.
    consolidated["cluster_kmeans"] = kmeans.predict(dataset).astype(str)
    logistic_features = logistic_metadata["numeric_features"] + [
        logistic_metadata["cluster_feature"]
    ]
    consolidated["probabilidade_deterioracao"] = logistic.predict_proba(
        consolidated[logistic_features]
    )[:, 1]
    consolidated["nivel_risco"] = risk_levels(
        consolidated["probabilidade_deterioracao"].to_numpy(),
        logistic_metadata["moderate_threshold"],
        logistic_metadata["high_threshold"],
    )
    consolidated["margem_fcf_prevista"] = forecaster.predict_margin(dataset)
    consolidated["fluxo_caixa_livre_previsto_usd_m"] = forecaster.predict(dataset)
    consolidated["tipo_predicao"] = consolidation_status(consolidated["split_ml"])

    target = "target_fluxo_caixa_livre_proximo_periodo_usd_m"
    if target in consolidated:
        consolidated["margem_fcf_real"] = consolidated[target].divide(
            consolidated["receita_liquida_usd_m"].replace(0, np.nan)
        )
        consolidated["residuo_fcf_usd_m"] = (
            consolidated[target] - consolidated["fluxo_caixa_livre_previsto_usd_m"]
        )
        consolidated["erro_absoluto_fcf_usd_m"] = consolidated[
            "residuo_fcf_usd_m"
        ].abs()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(args.output, index=False)
    metadata = {
        "artifact_version": "1.0.0",
        "dataset_file": args.input.name,
        "dataset_hash": file_hash(args.input),
        "row_count": len(consolidated),
        "output_file": args.output.name,
        "kmeans_artifact": kmeans_path.name,
        "kmeans_artifact_hash": file_hash(kmeans_path),
        "logistic_artifact": logistic_path.name,
        "logistic_artifact_hash": file_hash(logistic_path),
        "linear_artifact": linear_path.name,
        "linear_artifact_hash": file_hash(linear_path),
        "risk_thresholds": {
            "moderate": logistic_metadata["moderate_threshold"],
            "high": logistic_metadata["high_threshold"],
        },
        "prediction_note": (
            "Linhas de treino são previsões in-sample; validação e teste são holdout. "
            "Use somente holdout para estimar desempenho dos modelos."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path = args.output.with_name(
        "empresas_com_resultados_modelos_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Dataset consolidado: {args.output}")
    print(f"Empresas: {len(consolidated)}; colunas: {len(consolidated.columns)}")


if __name__ == "__main__":
    main()
