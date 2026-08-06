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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import settings
from data.dataset_loader import load_dataset
from features.financial_ratios import add_financial_ratios, safe_divide
from services.kmeans_clustering_service import QuantileClipper

TARGET = "target_deterioracao_financeira"
NUMERIC_FEATURES = (
    "crescimento_receita_pct",
    "taxa_crescimento_anual_pct",
    "margem_ebitda_pct",
    "margem_liquida_pct",
    "roa_pct",
    "debt_to_equity",
    "cobertura_juros",
    "liquidez_corrente",
    "liquidez_imediata",
    "fluxo_caixa_livre_usd_m",
    "capital_giro_usd_m",
    "indice_eficiencia_0_100",
)
CLUSTER_FEATURE = "cluster_kmeans"
# These variables are particularly long-tailed in the dataset.  The limits are
# learned inside the supervised pipeline from each training fold only, so the
# test partition never influences an outlier boundary.
CLIPPED_NUMERIC_FEATURES = (
    "debt_to_equity",
    "cobertura_juros",
    "fluxo_caixa_livre_usd_m",
    "capital_giro_usd_m",
)
RELATIVE_NUMERIC_FEATURES = tuple(
    feature
    for feature in NUMERIC_FEATURES
    if feature not in {"fluxo_caixa_livre_usd_m", "capital_giro_usd_m"}
) + ("margem_fluxo_caixa_livre_calculada", "capital_giro_sobre_receita")
GRID_LOGISTIC = {
    "logistic__C": [0.01, 0.1, 1.0, 10.0],
    "logistic__class_weight": [None, "balanced"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.raw_data_dir / "dataset_saude_financeira_8000_empresas.csv",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=100,
        help="Número de reamostragens para estabilidade dos coeficientes (padrão: 100).",
    )
    parser.add_argument(
        "--kmeans-artifact",
        type=Path,
        default=settings.models_dir / "kmeans_compact_v2.0.0.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.reports_dir / "risco_logistico",
    )
    return parser.parse_args()


def add_frozen_kmeans_cluster(
    dataset: pd.DataFrame, kmeans_pipeline: Pipeline
) -> pd.DataFrame:
    """Add a contextual cluster using the frozen K-Means; no fit occurs here."""
    enriched = dataset.copy()
    enriched[CLUSTER_FEATURE] = kmeans_pipeline.predict(dataset).astype(str)
    return enriched


def build_logistic_pipeline(
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES, *, include_cluster: bool = True
) -> Pipeline:
    """Build the supervised preprocessing and logistic regression pipeline.

    The numeric scaler is intentionally separate from K-Means: the logistic model
    uses a different feature set. The K-Means transformation is reused only to
    assign the cluster context; its eight-dimensional vector is not the logistic
    input specified for this stage.
    """
    transformers: list[tuple[str, Any, list[str]]] = [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
    ]
    if include_cluster:
        transformers.append(
            # drop='first' makes one cluster the reference category. Coefficients
            # for the remaining clusters are interpreted relative to that reference.
            (
                "cluster",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
                [CLUSTER_FEATURE],
            )
        )
    preprocessor = ColumnTransformer(
        transformers,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            # Clipping precedes imputation/scaling: the median and standard
            # deviation should describe the non-extreme part of the train data.
            (
                "outlier_clipper",
                QuantileClipper(
                    columns=tuple(
                        column for column in CLIPPED_NUMERIC_FEATURES if column in numeric_features
                    )
                ),
            ),
            ("preprocessor", preprocessor),
            ("logistic", LogisticRegression(max_iter=2_000, solver="liblinear")),
        ]
    )


def add_relative_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create size-neutral alternatives used only in the sensitivity comparison."""
    enriched = add_financial_ratios(dataset)
    enriched["capital_giro_sobre_receita"] = safe_divide(
        enriched["capital_giro_usd_m"], enriched["receita_liquida_usd_m"]
    )
    return enriched.replace([np.inf, -np.inf], np.nan)


def select_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, beta: float
) -> float:
    """Select a validation-only threshold maximizing F-beta for the positive class."""
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = [
        fbeta_score(y_true, probabilities >= threshold, beta=beta, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def evaluate_partition(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_sim": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_sim": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_sim": float(f1_score(y_true, predicted, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "classification_report": classification_report(
            y_true,
            predicted,
            target_names=["Não", "Sim"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[False, True]),
    }


def train_risk_model(
    dataset: pd.DataFrame,
    kmeans_pipeline: Pipeline,
    random_state: int = 42,
    bootstrap_iterations: int = 100,
) -> dict[str, Any]:
    """Train on treino, choose threshold on validação, then evaluate test once."""
    required = set(NUMERIC_FEATURES + (TARGET, "split_ml"))
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    enriched = add_frozen_kmeans_cluster(dataset, kmeans_pipeline)
    splits = {
        name: enriched.loc[enriched["split_ml"] == name].copy()
        for name in ("treino", "validacao", "teste")
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("treino, validacao and teste must all contain records.")

    x_train = splits["treino"][list(NUMERIC_FEATURES) + [CLUSTER_FEATURE]]
    y_train = splits["treino"][TARGET].eq("Sim").to_numpy()
    model = build_logistic_pipeline()
    min_class_size = int(pd.Series(y_train).value_counts().min())
    search = GridSearchCV(
        model,
        GRID_LOGISTIC,
        # PR-AUC avalia o ranking de risco sem escolher um limiar arbitrário.
        # A política operacional de corte é definida depois, só na validação.
        scoring="average_precision",
        cv=StratifiedKFold(
            n_splits=min(5, min_class_size), shuffle=True, random_state=random_state
        ),
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    # class_weight='balanced' pode melhorar atenção à classe rara, mas desloca as
    # probabilidades. A calibração sigmoide aprende esse ajuste dentro do treino.
    calibrated_model = calibrate_model(
        search.best_estimator_, x_train, y_train, random_state
    )

    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for name in ("validacao", "teste"):
        frame = splits[name]
        features = frame[list(NUMERIC_FEATURES) + [CLUSTER_FEATURE]]
        probabilities[name] = calibrated_model.predict_proba(features)[:, 1]
        targets[name] = frame[TARGET].eq("Sim").to_numpy()

    # F1 positive sets the binary decision threshold. F2 favors recall and becomes
    # the lower boundary for the moderate-risk band, avoiding arbitrary 0.40/0.65.
    high_threshold = select_threshold(
        targets["validacao"], probabilities["validacao"], beta=1.0
    )
    moderate_threshold = min(
        high_threshold,
        select_threshold(targets["validacao"], probabilities["validacao"], beta=2.0),
    )
    metrics = {
        name: evaluate_partition(targets[name], probabilities[name], high_threshold)
        for name in ("validacao", "teste")
    }
    test_predictions = splits["teste"][
        ["id_empresa", "nome_empresa_ficticio", CLUSTER_FEATURE]
    ].copy()
    test_predictions["probabilidade_deterioracao"] = probabilities["teste"]
    test_predictions["risco"] = risk_levels(
        probabilities["teste"], moderate_threshold, high_threshold
    )
    band_table, band_recall = risk_band_validation(
        probabilities["teste"], targets["teste"], moderate_threshold, high_threshold
    )
    specification_comparison, no_cluster_coefficients = compare_model_specifications(
        splits["treino"],
        splits["teste"],
        y_train,
        targets["teste"],
        search.best_params_,
        random_state,
    )
    return {
        "model": calibrated_model,
        "coefficient_model": search.best_estimator_,
        "enriched": enriched,
        "metrics": metrics,
        "probabilities": probabilities,
        "targets": targets,
        "test_predictions": test_predictions,
        "best_parameters": search.best_params_,
        "cross_validation_pr_auc": float(search.best_score_),
        "high_threshold": high_threshold,
        "moderate_threshold": moderate_threshold,
        "coefficients": coefficient_table(search.best_estimator_),
        "correlation": x_train.loc[:, list(NUMERIC_FEATURES)].corr(),
        "vif": calculate_vif(x_train.loc[:, list(NUMERIC_FEATURES)]),
        "bootstrap": bootstrap_coefficient_stability(
            search.best_estimator_,
            x_train,
            y_train,
            bootstrap_iterations,
            random_state,
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "specification_comparison": specification_comparison,
        "no_cluster_coefficients": no_cluster_coefficients,
        "risk_bands": band_table,
        "risk_band_recall": band_recall,
    }


def risk_levels(probabilities: np.ndarray, moderate: float, high: float) -> np.ndarray:
    """Apply validation-derived risk bands; high-risk uses the F1 decision threshold."""
    return np.select(
        [probabilities >= high, probabilities >= moderate],
        ["Alto", "Moderado"],
        default="Baixo",
    )


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Return coefficients and odds ratios in the transformed feature order."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["logistic"].coef_[0]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["effect"] = np.where(table["coefficient"] >= 0, "aumenta", "reduz")
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)


def calculate_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Estimate VIF after median imputation; high values flag multicollinearity.

    VIF is the reciprocal of one minus R² from regressing a feature on the others.
    It does not prove a coefficient is wrong, but explains why conditional signs can
    become unstable when financial indicators carry overlapping information.
    """
    prepared = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(features), columns=features.columns
    )
    values: list[dict[str, float | str]] = []
    for column in prepared.columns:
        others = prepared.drop(columns=column)
        r_squared = LinearRegression().fit(others, prepared[column]).score(others, prepared[column])
        vif = float("inf") if r_squared >= 0.999999 else float(1 / (1 - r_squared))
        values.append({"feature": column, "vif": vif})
    return pd.DataFrame(values).sort_values("vif", ascending=False)


def calibrate_model(
    estimator: Pipeline, features: pd.DataFrame, target: np.ndarray, random_state: int
) -> CalibratedClassifierCV:
    """Fit sigmoid calibration with folds formed exclusively from the train split."""
    min_class_size = int(pd.Series(target).value_counts().min())
    calibration_cv = StratifiedKFold(
        n_splits=min(5, min_class_size), shuffle=True, random_state=random_state
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method="sigmoid", cv=calibration_cv
    )
    return calibrated.fit(features, target)


def bootstrap_coefficient_stability(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    """Measure whether conditional coefficient signs persist across train resamples.

    Each bootstrap preserves the number of positive and negative records, then fits
    the *same* pipeline again.  This is a stability diagnostic, not a new model
    selection procedure; therefore it never consumes validation or test records.
    """
    if iterations < 1:
        return pd.DataFrame(
            columns=["feature", "mean", "std", "p2_5", "p97_5", "same_sign_rate"]
        )
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(target == value) for value in (False, True)]
    samples: list[pd.Series] = []
    for _ in range(iterations):
        index = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        fitted = clone(estimator).fit(features.iloc[index], target[index])
        samples.append(coefficient_table(fitted).set_index("feature")["coefficient"])
    coefficients = pd.concat(samples, axis=1)
    average = coefficients.mean(axis=1)
    return pd.DataFrame(
        {
            "feature": coefficients.index,
            "mean": average,
            "std": coefficients.std(axis=1),
            "p2_5": coefficients.quantile(0.025, axis=1),
            "p97_5": coefficients.quantile(0.975, axis=1),
            "same_sign_rate": coefficients.mul(np.sign(average), axis=0)
            .ge(0)
            .mean(axis=1),
        }
    ).sort_values("same_sign_rate")


def compare_model_specifications(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_train: np.ndarray,
    target_test: np.ndarray,
    best_parameters: dict[str, Any],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare cluster context and size-sensitive monetary inputs on held-out test.

    The alternatives reuse the selected regularization settings.  They are intended
    as sensitivity analysis: their test scores reveal whether an interpretation is
    contingent on the cluster feature or on absolute company-size proxies.
    """
    relative_train = add_relative_features(train)
    relative_test = add_relative_features(test)
    specifications = [
        ("com_cluster_e_valores_absolutos", NUMERIC_FEATURES, True, train, test),
        ("sem_cluster", NUMERIC_FEATURES, False, train, test),
        (
            "com_cluster_e_indicadores_relativos",
            RELATIVE_NUMERIC_FEATURES,
            True,
            relative_train,
            relative_test,
        ),
    ]
    metrics: list[dict[str, Any]] = []
    no_cluster_coefficients = pd.DataFrame()
    for name, numeric_features, include_cluster, train_frame, test_frame in specifications:
        columns = list(numeric_features) + ([CLUSTER_FEATURE] if include_cluster else [])
        estimator = build_logistic_pipeline(numeric_features, include_cluster=include_cluster)
        estimator.set_params(**best_parameters)
        calibrated = calibrate_model(
            estimator, train_frame[columns], target_train, random_state
        )
        probabilities = calibrated.predict_proba(test_frame[columns])[:, 1]
        metrics.append(
            {
                "especificacao": name,
                "roc_auc_teste": float(roc_auc_score(target_test, probabilities)),
                "pr_auc_teste": float(average_precision_score(target_test, probabilities)),
            }
        )
        if name == "sem_cluster":
            # Fit once on all training observations to expose its coefficients.
            uncalibrated = clone(estimator).fit(train_frame[columns], target_train)
            no_cluster_coefficients = coefficient_table(uncalibrated)
    return pd.DataFrame(metrics), no_cluster_coefficients


def risk_band_validation(
    probabilities: np.ndarray, target: np.ndarray, moderate: float, high: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Validate whether validation-defined low/moderate/high bands are monotonic in test."""
    levels = risk_levels(probabilities, moderate, high)
    frame = pd.DataFrame({"risco": levels, "deteriorou": target})
    overall_rate = float(frame["deteriorou"].mean())
    ordered = ["Baixo", "Moderado", "Alto"]
    table = frame.groupby("risco", observed=False)["deteriorou"].agg(["size", "sum", "mean"])
    table = table.reindex(ordered, fill_value=0).reset_index()
    table.columns = ["nivel", "empresas", "deterioraram", "taxa_observada"]
    table["lift"] = np.where(overall_rate > 0, table["taxa_observada"] / overall_rate, np.nan)
    high_only = levels == "Alto"
    monitored = levels != "Baixo"
    recall = {
        "recall_alto": float(target[high_only].sum() / target.sum()) if target.sum() else 0.0,
        "recall_moderado_mais_alto": float(target[monitored].sum() / target.sum()) if target.sum() else 0.0,
        "monotonic_rates": bool(table["taxa_observada"].is_monotonic_increasing),
    }
    return table, recall


def export_reports(
    result: dict[str, Any],
    output_dir: Path,
    *,
    kmeans_artifact_path: Path,
    input_dataset_path: Path,
) -> None:
    """Export all probability, classification, coefficient and visual reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result["test_predictions"].to_csv(output_dir / "predicoes_teste.csv", index=False)
    result["coefficients"].to_csv(
        output_dir / "coeficientes_odds_ratios.csv", index=False
    )
    result["correlation"].to_csv(output_dir / "matriz_correlacao.csv")
    result["vif"].to_csv(output_dir / "vif.csv", index=False)
    result["bootstrap"].to_csv(
        output_dir / "estabilidade_coeficientes_bootstrap.csv", index=False
    )
    result["specification_comparison"].to_csv(
        output_dir / "comparacao_especificacoes.csv", index=False
    )
    result["no_cluster_coefficients"].to_csv(
        output_dir / "coeficientes_sem_cluster.csv", index=False
    )
    result["risk_bands"].to_csv(output_dir / "validacao_faixas_risco_teste.csv", index=False)
    model_path = settings.models_dir / "regressao_logistica_risco_v1.0.0.joblib"
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], model_path)
    cluster_encoder = result["coefficient_model"].named_steps["preprocessor"].named_transformers_["cluster"]
    summary: dict[str, Any] = {
        "best_parameters": result["best_parameters"],
        "cross_validation_pr_auc": result["cross_validation_pr_auc"],
        "thresholds": {
            "moderate": result["moderate_threshold"],
            "high": result["high_threshold"],
        },
        "risk_band_recall": result["risk_band_recall"],
        "bootstrap_iterations": result["bootstrap_iterations"],
        "probability_note": "Probabilidade estimada calibrada por CV; não é probabilidade atuarial exata.",
        "interpretation": interpret_results(result),
    }
    for name, metrics in result["metrics"].items():
        summary[name] = {
            key: value
            for key, value in metrics.items()
            if key not in {"classification_report", "confusion_matrix"}
        }
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Real Não", "Real Sim"],
            columns=["Previsto Não", "Previsto Sim"],
        ).to_csv(output_dir / f"matriz_confusao_{name}.csv")
        (output_dir / f"classification_report_{name}.json").write_text(
            json.dumps(metrics["classification_report"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "resumo_risco_logistico.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "artifact_version": "1.0.0",
        "target": TARGET,
        "artifact_file": model_path.name,
        "kmeans_artifact": kmeans_artifact_path.name,
        "kmeans_artifact_hash": _file_hash(kmeans_artifact_path),
        "dataset_file": input_dataset_path.name,
        "dataset_hash": _file_hash(input_dataset_path),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "cluster_reference": str(cluster_encoder.categories_[0][0]),
        "moderate_threshold": result["moderate_threshold"],
        "high_threshold": result["high_threshold"],
        "numeric_features": list(NUMERIC_FEATURES),
        "cluster_feature": CLUSTER_FEATURE,
        "test_metrics": summary["teste"],
    }
    (settings.models_dir / "regressao_logistica_risco_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_plots(result, output_dir)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_plots(result: dict[str, Any], output_dir: Path) -> None:
    y_test, probabilities = result["targets"]["teste"], result["probabilities"]["teste"]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10)
    plots = [
        (fpr, tpr, "Taxa de falso positivo", "Recall", "Curva ROC", "roc.png"),
        (
            recall,
            precision,
            "Recall",
            "Precision",
            "Curva Precision-Recall",
            "precision_recall.png",
        ),
        (
            predicted,
            observed,
            "Probabilidade prevista",
            "Frequência observada",
            "Curva de calibração",
            "calibracao.png",
        ),
    ]
    for x, y, x_label, y_label, title, filename in plots:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(x, y, marker="o")
        if filename == "calibracao.png":
            axis.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="gray",
                label="calibração perfeita",
            )
            axis.legend()
        axis.set(title=title, xlabel=x_label, ylabel=y_label)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)
    # A tabela de faixas já contém a evidência numérica. Este gráfico apresenta a
    # mesma taxa observada com o suporte de cada faixa, evitando interpretar uma
    # barra alta de um grupo pequeno sem conhecer sua quantidade de empresas.
    bands = result["risk_bands"].copy()
    colors = {"Baixo": "#4C78A8", "Moderado": "#F2A541", "Alto": "#C94C4C"}
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        bands["nivel"],
        bands["taxa_observada"],
        color=[colors.get(level, "#808080") for level in bands["nivel"]],
    )
    axis.set(
        title="Taxa observada de deterioração por faixa de risco",
        xlabel="Faixa de risco",
        ylabel="Taxa observada de deterioração",
        ylim=(0, 1),
    )
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    for bar, row in zip(bars, bands.itertuples(index=False), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{row.taxa_observada:.1%}\n{row.deterioraram}/{row.empresas} empresas",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output_dir / "taxa_deterioracao_por_faixa_risco.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(probabilities, bins=20)
    axis.set(
        title="Distribuição das probabilidades no teste",
        xlabel="Probabilidade de deterioração",
        ylabel="Empresas",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "distribuicao_probabilidades.png", dpi=150)
    plt.close(figure)
    coefficients = result["coefficients"].sort_values("coefficient")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(coefficients["feature"], coefficients["coefficient"])
    axis.set(
        title="Coeficientes da regressão logística", xlabel="Coeficiente padronizado"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "coeficientes.png", dpi=150)
    plt.close(figure)

def interpret_results(result: dict[str, Any]) -> list[str]:
    """Describe recall trade-offs and the strongest logistic associations cautiously."""
    test = result["metrics"]["teste"]
    matrix = test["confusion_matrix"]
    false_negatives, false_positives = int(matrix[1, 0]), int(matrix[0, 1])
    messages = [
        f"No teste, o recall de 'Sim' foi {test['recall_sim']:.1%}: "
        f"{false_negatives} deteriorações não foram sinalizadas.",
        f"O limiar {result['high_threshold']:.3f} gera {false_positives} "
        "falsos positivos; reduzi-lo aumenta recall e alertas.",
        f"Brier Score de {test['brier_score']:.4f}; valores menores indicam "
        "probabilidades mais bem calibradas.",
    ]
    for row in result["coefficients"].head(3).itertuples(index=False):
        if row.feature.startswith("cluster_kmeans_"):
            cluster = row.feature.removeprefix("cluster_kmeans_")
            direction = "menor" if row.effect == "reduz" else "maior"
            messages.append(
                f"Pertencer ao Cluster {cluster} está associado a risco relativo "
                f"{direction} versus o cluster de referência "
                f"(odds ratio {row.odds_ratio:.2f})."
            )
            continue
        direction = "maior" if row.effect == "aumenta" else "menor"
        messages.append(
            f"{row.feature}: valor {direction} está associado a maior risco "
            f"relativo (odds ratio {row.odds_ratio:.2f})."
        )
    return messages


def print_summary(result: dict[str, Any]) -> None:
    for name in ("validacao", "teste"):
        metrics = result["metrics"][name]
        print(
            f"\n{name.upper()} — ROC-AUC {metrics['roc_auc']:.3f}; "
            f"PR-AUC {metrics['pr_auc']:.3f}; Recall Sim {metrics['recall_sim']:.3f}"
        )
    print("Limiar alto (F1 validação):", f"{result['high_threshold']:.3f}")
 

def main() -> None:
    args = parse_args()
    if not args.kmeans_artifact.is_file():
        raise FileNotFoundError(
            "K-Means artifact not found. Run the compact K-Means script first."
        )
    dataset = load_dataset(args.input)
    result = train_risk_model(
        dataset,
        joblib.load(args.kmeans_artifact),
        bootstrap_iterations=args.bootstrap_iterations,
    )
    export_reports(
        result,
        args.output_dir,
        kmeans_artifact_path=args.kmeans_artifact,
        input_dataset_path=args.input,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
