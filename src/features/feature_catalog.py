from domain.dataset_schema import CATEGORICAL_COLUMNS, FINANCIAL_COLUMNS, TARGET_COLUMNS

IDENTIFIER_AND_SPLIT_COLUMNS = ("id_empresa", "nome_empresa_ficticio", "split_ml")
SYNTHETIC_RISK_COLUMNS = (
    "indice_risco_financeiro_0_100",
    "indice_saude_financeira_0_100",
    "score_financeiro_geral_0_100",
)
PROHIBITED_FEATURE_COLUMNS = (
    TARGET_COLUMNS + SYNTHETIC_RISK_COLUMNS + IDENTIFIER_AND_SPLIT_COLUMNS
)

BASE_NUMERIC_FEATURES = tuple(
    column for column in FINANCIAL_COLUMNS if column not in SYNTHETIC_RISK_COLUMNS
) + ("ano_referencia",)
BASE_CATEGORICAL_FEATURES = CATEGORICAL_COLUMNS
RATIO_FEATURES = (
    "margem_bruta_calculada",
    "margem_ebitda_calculada",
    "margem_liquida_calculada",
    "divida_sobre_ativos",
    "caixa_sobre_passivos_circulantes",
    "fluxo_operacional_sobre_receita",
)

CLASSIFICATION_FEATURES = (
    BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES + RATIO_FEATURES
)
REGRESSION_FEATURES = CLASSIFICATION_FEATURES
CLUSTERING_FEATURES = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES + RATIO_FEATURES
INTERPRETABLE_TREE_FEATURES = CLASSIFICATION_FEATURES
