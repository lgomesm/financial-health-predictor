import json

import pandas as pd
import streamlit as st

from config.settings import settings

st.title("Risco financeiro")
st.info(
    "A probabilidade de deterioração é uma previsão estatística, "
    "não uma relação causal."
)

reports_dir = settings.reports_dir / "risco_logistico"
summary_path = reports_dir / "resumo_risco_logistico.json"
if not summary_path.exists():
    st.warning(
        "Execute prever_deterioracao_logistica.py para gerar a análise de risco."
    )
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
test_metrics = summary["teste"]
columns = st.columns(4)
columns[0].metric("ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
columns[1].metric("PR-AUC", f"{test_metrics['pr_auc']:.3f}")
columns[2].metric("Brier Score", f"{test_metrics['brier_score']:.3f}")
columns[3].metric("Recall — Sim", f"{test_metrics['recall_sim']:.1%}")

st.caption(
    f"Limiar alto de risco: {summary['thresholds']['high']:.3f}. "
    f"Limiar moderado: {summary['thresholds']['moderate']:.3f}."
)
st.caption(summary["probability_note"])
for interpretation in summary["interpretation"]:
    st.write(f"- {interpretation}")

st.subheader("Matriz de confusão")
st.dataframe(
    pd.read_csv(reports_dir / "matriz_confusao_teste.csv", index_col=0),
    use_container_width=True,
)

st.subheader("Coeficientes e Odds Ratios")
st.dataframe(
    pd.read_csv(reports_dir / "coeficientes_odds_ratios.csv"),
    use_container_width=True,
)

st.subheader("Empresas por nível de risco")
predictions = pd.read_csv(reports_dir / "predicoes_teste.csv")
st.dataframe(
    predictions.sort_values("probabilidade_deterioracao", ascending=False),
    use_container_width=True,
)

bands_path = reports_dir / "validacao_faixas_risco_teste.csv"
if bands_path.exists():
    st.subheader("Validação empírica das faixas de risco")
    st.dataframe(pd.read_csv(bands_path), use_container_width=True)
    band_recall = summary["risk_band_recall"]
    st.caption(
        f"Recall somente alto: {band_recall['recall_alto']:.1%}; "
        f"moderado + alto: {band_recall['recall_moderado_mais_alto']:.1%}."
    )
    bands_chart = reports_dir / "taxa_deterioracao_por_faixa_risco.png"
    if bands_chart.exists():
        st.image(str(bands_chart), use_container_width=True)

with st.expander("Diagnóstico de multicolinearidade"):
    correlation_path = reports_dir / "matriz_correlacao.csv"
    vif_path = reports_dir / "vif.csv"
    if correlation_path.exists() and vif_path.exists():
        st.dataframe(pd.read_csv(vif_path), use_container_width=True)
        st.dataframe(pd.read_csv(correlation_path, index_col=0), use_container_width=True)

with st.expander("Robustez dos coeficientes e sensibilidade do modelo"):
    comparison_path = reports_dir / "comparacao_especificacoes.csv"
    stability_path = reports_dir / "estabilidade_coeficientes_bootstrap.csv"
    no_cluster_path = reports_dir / "coeficientes_sem_cluster.csv"
    if comparison_path.exists():
        st.caption(
            "As comparações são análises de sensibilidade no teste: coeficientes "
            "descrevem associações condicionais, não relações causais."
        )
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
    if stability_path.exists():
        st.dataframe(pd.read_csv(stability_path), use_container_width=True)
    if no_cluster_path.exists():
        st.caption("Coeficientes da versão sem o contexto de cluster K-Means.")
        st.dataframe(pd.read_csv(no_cluster_path), use_container_width=True)

st.subheader("Gráficos de probabilidade e interpretação")
for filename in (
    "roc.png",
    "precision_recall.png",
    "calibracao.png",
    "distribuicao_probabilidades.png",
    "coeficientes.png",
):
    image_path = reports_dir / filename
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
