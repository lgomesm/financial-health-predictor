from dataclasses import dataclass

IDENTIFIER_COLUMNS = ("id_empresa", "nome_empresa_ficticio")

METADATA_COLUMNS = ("ano_referencia", "split_ml")

CATEGORICAL_COLUMNS = (
    "setor_economico",
    "subsetor",
    "pais",
    "regiao",
    "porte_empresa",
    "estagio_maturidade",
)

FINANCIAL_COLUMNS = (
    "receita_bruta_usd_m",
    "receita_liquida_usd_m",
    "receita_liquida_t_1_usd_m",
    "receita_liquida_t_2_usd_m",
    "receita_recorrente_usd_m",
    "receita_internacional_usd_m",
    "crescimento_receita_pct",
    "taxa_crescimento_anual_pct",
    "lucro_bruto_usd_m",
    "ebitda_usd_m",
    "margem_ebitda_pct",
    "lucro_operacional_usd_m",
    "margem_operacional_pct",
    "lucro_liquido_usd_m",
    "margem_liquida_pct",
    "margem_bruta_pct",
    "roa_pct",
    "roe_pct",
    "roi_pct",
    "ativos_totais_usd_m",
    "passivos_totais_usd_m",
    "divida_curto_prazo_usd_m",
    "divida_longo_prazo_usd_m",
    "divida_total_usd_m",
    "patrimonio_liquido_usd_m",
    "debt_to_equity",
    "indice_alavancagem",
    "cobertura_juros",
    "ativos_circulantes_usd_m",
    "passivos_circulantes_usd_m",
    "caixa_usd_m",
    "equivalentes_caixa_usd_m",
    "capital_giro_usd_m",
    "liquidez_corrente",
    "liquidez_seca",
    "liquidez_imediata",
    "fluxo_operacional_usd_m",
    "fluxo_investimentos_usd_m",
    "fluxo_financiamentos_usd_m",
    "fluxo_caixa_livre_usd_m",
    "fluxo_caixa_livre_t_1_usd_m",
    "variacao_caixa_usd_m",
    "custos_operacionais_usd_m",
    "despesas_administrativas_usd_m",
    "despesas_financeiras_usd_m",
    "investimentos_capex_usd_m",
    "opex_usd_m",
    "giro_ativo",
    "giro_estoque",
    "prazo_medio_recebimento_dias",
    "prazo_medio_pagamento_dias",
    "valor_mercado_estimado_usd_m",
    "numero_funcionarios",
    "market_share_estimado_pct",
    "participacao_internacional_pct",
    "indice_solvencia",
    "indice_cobertura",
    "indice_eficiencia_0_100",
    "indice_risco_financeiro_0_100",
    "indice_saude_financeira_0_100",
    "score_financeiro_geral_0_100",
)

TARGET_COLUMNS = (
    "target_deterioracao_financeira",
    "target_probabilidade_deterioracao_pct",
    "target_classe_risco",
    "target_faixa_saude_financeira",
    "target_probabilidade_insolvencia_pct",
    "target_tende_crescer_proximo_periodo",
    "target_fluxo_caixa_livre_proximo_periodo_usd_m",
    "target_crescimento_receita_proximo_periodo_pct",
)

NUMERIC_COLUMNS = FINANCIAL_COLUMNS + (
    "ano_referencia",
    "target_probabilidade_deterioracao_pct",
    "target_probabilidade_insolvencia_pct",
    "target_fluxo_caixa_livre_proximo_periodo_usd_m",
    "target_crescimento_receita_proximo_periodo_pct",
)

TARGET_ALLOWED_VALUES = {
    "target_deterioracao_financeira": frozenset({"Sim", "Não"}),
    "target_classe_risco": frozenset({"Baixo", "Médio", "Alto"}),
    "target_faixa_saude_financeira": frozenset(
        {"Crítica", "Atenção", "Boa", "Excelente"}
    ),
    "target_tende_crescer_proximo_periodo": frozenset({"Sim", "Não"}),
}


@dataclass(frozen=True, slots=True)
class FinancialDatasetSchema:
    identifiers: tuple[str, ...] = IDENTIFIER_COLUMNS
    metadata: tuple[str, ...] = METADATA_COLUMNS
    categoricals: tuple[str, ...] = CATEGORICAL_COLUMNS
    financials: tuple[str, ...] = FINANCIAL_COLUMNS
    targets: tuple[str, ...] = TARGET_COLUMNS

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the complete column contract in its expected order"""
        return (
            self.identifiers
            + self.categoricals
            + self.metadata
            + self.financials
            + self.targets
        )

FINANCIAL_DATASET_SCHEMA = FinancialDatasetSchema()