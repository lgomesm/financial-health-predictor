from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float | None
    smape: float | None
    wape: float | None
    mape_coverage: float
    median_absolute_error: float
    p75_absolute_error: float
    p90_absolute_error: float
    p95_absolute_error: float


def calculate_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    """Calculate absolute and robust percentage errors for a regression result.

    MAPE is not meaningful when the actual FCF is zero or nearly zero.  Those rows
    are excluded only from MAPE and their share is reported; MAE/RMSE/R² still use
    every record. sMAPE is reported alongside it because its symmetric denominator
    is less dominated by small actual values.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    usable_for_mape = np.abs(actual) > 1e-6
    mape = (
        float(np.mean(np.abs((actual[usable_for_mape] - predicted[usable_for_mape]) / actual[usable_for_mape])))
        if usable_for_mape.any()
        else None
    )
    denominator = np.abs(actual) + np.abs(predicted)
    usable_for_smape = denominator > 1e-6
    smape = (
        float(np.mean(2 * np.abs(actual[usable_for_smape] - predicted[usable_for_smape]) / denominator[usable_for_smape]))
        if usable_for_smape.any()
        else None
    )
    absolute_errors = np.abs(actual - predicted)
    total_actual = np.abs(actual).sum()
    return RegressionMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(mean_squared_error(actual, predicted) ** 0.5),
        r2=float(r2_score(actual, predicted)),
        mape=mape,
        smape=smape,
        wape=float(absolute_errors.sum() / total_actual) if total_actual > 1e-6 else None,
        mape_coverage=float(usable_for_mape.mean()),
        median_absolute_error=float(np.quantile(absolute_errors, 0.5)),
        p75_absolute_error=float(np.quantile(absolute_errors, 0.75)),
        p90_absolute_error=float(np.quantile(absolute_errors, 0.9)),
        p95_absolute_error=float(np.quantile(absolute_errors, 0.95)),
    )


def residual_report(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Return actual, predicted, and residual values for inspection or plotting."""
    return pd.DataFrame({"actual": actual, "predicted": predicted, "residual": actual - predicted})


def metrics_by_segment(dataset: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, segment: str) -> dict[str, RegressionMetrics]:
    """Calculate comparable metrics by company segment."""
    return {str(value): calculate_regression_metrics(actual[mask], predicted[mask]) for value in dataset[segment].unique() if (mask := dataset[segment].eq(value).to_numpy()).sum() >= 2}
