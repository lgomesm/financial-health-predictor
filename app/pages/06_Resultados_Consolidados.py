"""Exploração amigável das saídas integradas dos modelos já treinados."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import settings

CONSOLIDATED_PATH = (
    settings.processed_data_dir / "empresas_com_resultados_modelos.csv"
)


@st.cache_data
def load_consolidated_data(path: str) -> pd.DataFrame:
    """Lê o CSV consolidado uma vez por alteração do arquivo na aplicação."""
    return pd.read_csv(path)


def to_csv_bytes(data: pd.DataFrame) -> bytes:
    """Prepara um recorte filtrado para download sem gravar outro arquivo no disco."""
    return data.to_csv(index=False).encode("utf-8-sig")


def format_company(option: str | None, data: pd.DataFrame) -> str:
    """Exibe nome e identificador, preservando o identificador como chave da seleção."""
    if option is None:
        return "Selecione uma empresa"
    company = data.loc[data["id_empresa"].eq(option)].iloc[0]
    return f"{company['nome_empresa_ficticio']} ({option})"


st.title("Resultados consolidados")
st.caption(
    "Consulte, filtre e compare as saídas do K-Means, da regressão logística "
    "e da previsão de fluxo de caixa em um único dataset."
)

if not CONSOLIDATED_PATH.exists():
    st.warning(
        "Dataset consolidado ainda não encontrado. Execute "
        "consolidar_resultados_modelos.py para gerá-lo."
    )
    st.stop()

data = load_consolidated_data(str(CONSOLIDATED_PATH))
required_columns = {
    "id_empresa",
    "nome_empresa_ficticio",
    "split_ml",
    "cluster_kmeans",
    "probabilidade_deterioracao",
    "nivel_risco",
    "fluxo_caixa_livre_previsto_usd_m",
}
missing_columns = sorted(required_columns - set(data.columns))
if missing_columns:
    st.error(
        "O CSV consolidado não possui as colunas necessárias: "
        + ", ".join(missing_columns)
    )
    st.stop()

st.info(
    "As saídas exibidas foram geradas por artefatos já treinados; esta página "
    "apenas consulta seus resultados e não ajusta nenhum modelo."
)

with st.sidebar:
    st.header("Filtros")
    selected_splits = st.multiselect(
        "Partição",
        options=sorted(data["split_ml"].dropna().unique()),
        default=sorted(data["split_ml"].dropna().unique()),
    )
    selected_clusters = st.multiselect(
        "Cluster K-Means",
        options=sorted(data["cluster_kmeans"].dropna().unique()),
        default=sorted(data["cluster_kmeans"].dropna().unique()),
    )
    risk_order = ["Baixo", "Moderado", "Alto"]
    available_risks = [risk for risk in risk_order if risk in set(data["nivel_risco"])]
    selected_risks = st.multiselect(
        "Nível de risco",
        options=available_risks,
        default=available_risks,
    )
    selected_sizes = st.multiselect(
        "Porte",
        options=sorted(data["porte_empresa"].dropna().unique()),
        default=sorted(data["porte_empresa"].dropna().unique()),
    )
    selected_sectors = st.multiselect(
        "Setor",
        options=sorted(data["setor_economico"].dropna().unique()),
        default=sorted(data["setor_economico"].dropna().unique()),
    )
    company_query = st.text_input("Buscar por empresa ou ID")

filtered = data.loc[
    data["split_ml"].isin(selected_splits)
    & data["cluster_kmeans"].isin(selected_clusters)
    & data["nivel_risco"].isin(selected_risks)
    & data["porte_empresa"].isin(selected_sizes)
    & data["setor_economico"].isin(selected_sectors)
].copy()

if company_query:
    query = company_query.casefold()
    matches_name = filtered["nome_empresa_ficticio"].str.casefold().str.contains(
        query, na=False
    )
    matches_id = filtered["id_empresa"].str.casefold().str.contains(query, na=False)
    filtered = filtered.loc[matches_name | matches_id].copy()

if filtered.empty:
    st.warning("Nenhuma empresa corresponde aos filtros selecionados.")
    st.stop()

high_risk_share = filtered["nivel_risco"].eq("Alto").mean()
negative_fcf_share = filtered["fluxo_caixa_livre_previsto_usd_m"].lt(0).mean()
metrics = st.columns(4)
metrics[0].metric("Empresas no recorte", f"{len(filtered):,}".replace(",", "."))
metrics[1].metric("Risco alto", f"{high_risk_share:.1%}")
metrics[2].metric("FCF futuro negativo", f"{negative_fcf_share:.1%}")
metrics[3].metric(
    "Probabilidade média de deterioração",
    f"{filtered['probabilidade_deterioracao'].mean():.1%}",
)

overview_tab, companies_tab = st.tabs(["Visão geral", "Empresas"])

with overview_tab:
    chart_columns = st.columns(2)
    risk_counts = (
        filtered.groupby("nivel_risco", observed=True)
        .size()
        .reindex(risk_order, fill_value=0)
        .rename_axis("nivel_risco")
        .reset_index(name="empresas")
    )
    chart_columns[0].plotly_chart(
        px.bar(
            risk_counts,
            x="nivel_risco",
            y="empresas",
            color="nivel_risco",
            category_orders={"nivel_risco": risk_order},
            color_discrete_map={
                "Baixo": "#4C78A8",
                "Moderado": "#F2A541",
                "Alto": "#C94C4C",
            },
            title="Empresas por nível de risco",
            labels={"nivel_risco": "Nível de risco", "empresas": "Empresas"},
        ),
        use_container_width=True,
    )

    cluster_counts = (
        filtered.groupby("cluster_kmeans", observed=True)
        .size()
        .reset_index(name="empresas")
    )
    chart_columns[1].plotly_chart(
        px.bar(
            cluster_counts,
            x="cluster_kmeans",
            y="empresas",
            color="cluster_kmeans",
            title="Empresas por perfil financeiro",
            labels={"cluster_kmeans": "Cluster", "empresas": "Empresas"},
        ),
        use_container_width=True,
    )

    st.subheader("Risco estimado e previsão de fluxo de caixa")
    scatter_columns = [
        "id_empresa",
        "nome_empresa_ficticio",
        "porte_empresa",
        "setor_economico",
        "cluster_kmeans",
        "nivel_risco",
        "probabilidade_deterioracao",
        "fluxo_caixa_livre_previsto_usd_m",
    ]
    st.plotly_chart(
        px.scatter(
            filtered,
            x="probabilidade_deterioracao",
            y="fluxo_caixa_livre_previsto_usd_m",
            color="nivel_risco",
            hover_name="nome_empresa_ficticio",
            hover_data=scatter_columns,
            category_orders={"nivel_risco": risk_order},
            color_discrete_map={
                "Baixo": "#4C78A8",
                "Moderado": "#F2A541",
                "Alto": "#C94C4C",
            },
            labels={
                "probabilidade_deterioracao": "Probabilidade de deterioração",
                "fluxo_caixa_livre_previsto_usd_m": "FCF futuro previsto (USD mi)",
                "nivel_risco": "Nível de risco",
            },
            title="Probabilidade de deterioração versus FCF futuro previsto",
        ),
        use_container_width=True,
    )

with companies_tab:
    st.subheader("Consulta individual")
    company_ids = sorted(filtered["id_empresa"].astype(str).unique())
    selected_company_id = st.selectbox(
        "Escolha uma empresa no recorte filtrado",
        options=[None, *company_ids],
        format_func=lambda option: format_company(option, filtered),
    )

    if selected_company_id is not None:
        company = filtered.loc[filtered["id_empresa"].eq(selected_company_id)].iloc[0]
        st.markdown(f"### {company['nome_empresa_ficticio']}")
        st.caption(
            f"{company['setor_economico']} · {company['porte_empresa']} · "
            f"Partição: {company['split_ml']}"
        )

        model_metrics = st.columns(4)
        model_metrics[0].metric("Cluster", str(company["cluster_kmeans"]))
        model_metrics[1].metric(
            "Probabilidade de deterioração",
            f"{company['probabilidade_deterioracao']:.1%}",
        )
        model_metrics[2].metric("Nível de risco", company["nivel_risco"])
        model_metrics[3].metric(
            "FCF futuro previsto",
            f"USD {company['fluxo_caixa_livre_previsto_usd_m']:,.3f} mi",
        )

        indicators = [
            ("Crescimento da receita", "crescimento_receita_pct", "%"),
            ("Margem EBITDA", "margem_ebitda_pct", "%"),
            ("Margem líquida", "margem_liquida_pct", "%"),
            ("Debt-to-equity", "debt_to_equity", ""),
            ("Liquidez corrente", "liquidez_corrente", ""),
            ("Cobertura de juros", "cobertura_juros", ""),
            ("FCF atual", "fluxo_caixa_livre_usd_m", "USD mi"),
        ]
        indicator_rows = []
        for label, column, unit in indicators:
            if column in company.index:
                indicator_rows.append(
                    {"Indicador": label, "Valor": company[column], "Unidade": unit}
                )
        st.subheader("Indicadores financeiros selecionados")
        st.dataframe(pd.DataFrame(indicator_rows), use_container_width=True)

    st.subheader("Empresas do recorte")
    table_columns = [
        "id_empresa",
        "nome_empresa_ficticio",
        "split_ml",
        "porte_empresa",
        "setor_economico",
        "cluster_kmeans",
        "probabilidade_deterioracao",
        "nivel_risco",
        "fluxo_caixa_livre_previsto_usd_m",
    ]
    available_table_columns = [
        column for column in table_columns if column in filtered.columns
    ]
    table = filtered[available_table_columns].sort_values(
        "probabilidade_deterioracao", ascending=False
    )
    st.dataframe(table, use_container_width=True, height=420)
    st.download_button(
        "Baixar empresas filtradas (CSV)",
        data=to_csv_bytes(filtered),
        file_name="empresas_com_resultados_filtradas.csv",
        mime="text/csv",
    )

st.caption(
    "Linhas de treino representam previsões in-sample. Para avaliar desempenho, "
    "utilize as linhas de validação e teste, identificadas como holdout no campo "
    "tipo_predicao."
)
