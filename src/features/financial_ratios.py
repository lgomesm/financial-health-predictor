import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series, denominator: pd.Series, *, minimum_denominator: float = 1e-12
) -> pd.Series:
    valid_denominator = denominator.where(denominator.abs() > minimum_denominator)
    return numerator.divide(valid_denominator).replace([np.inf, -np.inf], np.nan)


def add_financial_ratios(dataset: pd.DataFrame) -> pd.DataFrame:
    enriched = dataset.copy()
    enriched["margem_bruta_calculada"] = safe_divide(
        enriched["lucro_bruto_usd_m"], enriched["receita_liquida_usd_m"]
    )
    enriched["margem_ebitda_calculada"] = safe_divide(
        enriched["ebitda_usd_m"], enriched["receita_liquida_usd_m"]
    )
    enriched["margem_liquida_calculada"] = safe_divide(
        enriched["lucro_liquido_usd_m"], enriched["receita_liquida_usd_m"]
    )
    enriched["divida_sobre_ativos"] = safe_divide(
        enriched["divida_total_usd_m"], enriched["ativos_totais_usd_m"]
    )
    enriched["caixa_sobre_passivos_circulantes"] = safe_divide(
        enriched["caixa_usd_m"], enriched["passivos_circulantes_usd_m"]
    )
    enriched["fluxo_operacional_sobre_receita"] = safe_divide(
        enriched["fluxo_operacional_usd_m"], enriched["receita_liquida_usd_m"]
    )
    enriched["margem_fluxo_caixa_livre_calculada"] = safe_divide(
        enriched["fluxo_caixa_livre_usd_m"], enriched["receita_liquida_usd_m"]
    )
    return enriched