from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.regression_metrics import RegressionMetrics, calculate_regression_metrics
from features.financial_ratios import safe_divide
from services.kmeans_clustering_service import QuantileClipper

TARGET = "target_fluxo_caixa_livre_proximo_periodo_usd_m"
TARGET_MARGIN = "target_margem_fcf_futura"
CLUSTER_FEATURE = "cluster_kmeans"

# The model consumes financial ratios rather than raw monetary amounts. This makes
# its learned relationship less dependent on company size. Revenue remains outside
# the estimator and is used only after prediction to reconstruct the dollar amount.
RAW_COLUMNS = (
    "receita_liquida_usd_m",
    "crescimento_receita_pct",
    "margem_ebitda_pct",
    "fluxo_caixa_livre_usd_m",
    "capital_giro_usd_m",
    "divida_total_usd_m",
)
NUMERIC_FEATURES = (
    "crescimento_receita_pct",
    "margem_ebitda_pct",
    "margem_fcf_atual",
    "capital_giro_sobre_receita",
    "divida_sobre_receita",
)
CLIPPED_FEATURES = NUMERIC_FEATURES


class CashflowFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create current/past ratios using only information available at prediction time."""

    def fit(self, data: pd.DataFrame, y: object = None) -> CashflowFeatureTransformer:
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = data.copy()
        for column in RAW_COLUMNS:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
        revenue = transformed["receita_liquida_usd_m"]
        transformed["margem_fcf_atual"] = safe_divide(
            transformed["fluxo_caixa_livre_usd_m"], revenue
        )
        transformed["capital_giro_sobre_receita"] = safe_divide(
            transformed["capital_giro_usd_m"], revenue
        )
        transformed["divida_sobre_receita"] = safe_divide(
            transformed["divida_total_usd_m"], revenue
        )
        return transformed.replace([np.inf, -np.inf], np.nan)


def build_cashflow_pipeline(regressor: object | None = None) -> Pipeline:
    """Build OLS with train-fitted clipping, imputation, scaling and cluster context."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
            (
                "cluster",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                [CLUSTER_FEATURE],
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("ratios", CashflowFeatureTransformer()),
            ("outlier_clipper", QuantileClipper(CLIPPED_FEATURES)),
            ("preprocessor", preprocessor),
            ("regression", regressor if regressor is not None else LinearRegression()),
        ]
    )


def add_target_margin(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create the future FCF margin target and remove non-computable observations."""
    enriched = dataset.copy()
    enriched[TARGET_MARGIN] = safe_divide(enriched[TARGET], enriched["receita_liquida_usd_m"])
    return enriched.dropna(subset=[TARGET_MARGIN, "receita_liquida_usd_m"])


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Expose serializable absolute-FCF metrics, including safe percentage metrics."""
    return asdict(calculate_regression_metrics(actual, predicted))


def coefficients_table(model: Pipeline) -> pd.DataFrame:
    """Return standardized OLS coefficients; they are conditional associations."""
    preprocessor = model.named_steps["preprocessor"]
    table = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient": model.named_steps["regression"].coef_,
        }
    )
    # Absolute coefficient is useful for sorting numeric standardized inputs, but
    # it is deliberately not labelled "importance": cluster dummies are relative
    # to the reference cluster and cannot be compared on that scale.
    table["absolute_coefficient"] = table["coefficient"].abs()
    table["direction"] = np.where(table["coefficient"] >= 0, "aumenta", "reduz")
    return table.sort_values("absolute_coefficient", ascending=False)


@dataclass(slots=True)
class CashflowForecaster:
    """Production contract: raw company data in, predicted FCF in USD millions out."""

    kmeans_pipeline: Pipeline
    regression_pipeline: Pipeline

    def predict_margin(self, raw_dataset: pd.DataFrame) -> np.ndarray:
        enriched = raw_dataset.copy()
        enriched[CLUSTER_FEATURE] = self.kmeans_pipeline.predict(raw_dataset).astype(str)
        return self.regression_pipeline.predict(enriched)

    def predict(self, raw_dataset: pd.DataFrame) -> np.ndarray:
        """Reconstruct dollar FCF from the predicted margin and current revenue."""
        revenue = pd.to_numeric(raw_dataset["receita_liquida_usd_m"], errors="coerce")
        return self.predict_margin(raw_dataset) * revenue.to_numpy(dtype=float)


def metrics_by_group(
    data: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, group: str
) -> pd.DataFrame:
    """Calculate metrics per segment without hiding sample sizes."""
    rows: list[dict[str, Any]] = []
    for value, subset in data.groupby(group, dropna=False):
        positions = subset.index.to_numpy()
        # The caller supplies a reset index, so positions map directly to arrays.
        if len(positions) < 2:
            continue
        metrics = evaluate_forecast(actual[positions], predicted[positions])
        rows.append({"grupo": str(value), "empresas": len(positions), **metrics})
    return pd.DataFrame(rows).sort_values("mae") if rows else pd.DataFrame()
