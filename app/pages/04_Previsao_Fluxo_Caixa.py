import json

import pandas as pd
import streamlit as st

from config.settings import settings

st.title("Previsão de fluxo de caixa")
st.info(
    "A projeção é quantitativa e condicional à manutenção da receita atual; "
    "não representa recomendação financeira."
)
reports_dir = settings.reports_dir / "fluxo_caixa_linear"
summary_path = reports_dir / "resumo_fluxo_caixa.json"
if not summary_path.exists():
    st.warning("Execute prever_fluxo_caixa_linear.py para gerar a análise.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
test = summary["metrics_fcf_reconstructed"]["teste"]
columns = st.columns(5)
columns[0].metric("MAE", f"USD {test['mae']:.2f} mi")
columns[1].metric("RMSE", f"USD {test['rmse']:.2f} mi")
columns[2].metric("R²", f"{test['r2']:.3f}")
columns[3].metric("WAPE", "n/a" if test["wape"] is None else f"{test['wape']:.1%}")
columns[4].metric("sMAPE", "n/a" if test["smape"] is None else f"{test['smape']:.1%}")
st.caption(
    "MAPE está disponível no CSV, mas não é métrica principal: FCF pode ser negativo "
    "ou próximo de zero. WAPE e sMAPE são mais estáveis nesse contexto."
)
st.subheader("Comparação com baselines")
baseline_path = reports_dir / "comparacao_baselines.csv"
if baseline_path.exists():
    st.dataframe(pd.read_csv(baseline_path), use_container_width=True)
st.caption(
    "O baseline de persistência prevê que o próximo FCF será igual ao FCF atual. "
    "A OLS só demonstra ganho prático se superar essa referência."
)
for text in summary["interpretation"]:
    st.write(f"- {text}")

predictions = pd.read_csv(reports_dir / "previsoes_teste.csv")
st.subheader("Previsões por empresa")
filter_columns = st.columns(3)
for label, column, widget in (
    ("Cluster", "cluster_kmeans", filter_columns[0]),
    ("Porte", "porte_empresa", filter_columns[1]),
    ("Nível de risco", "nivel_risco", filter_columns[2]),
):
    if column in predictions:
        selected = widget.multiselect(label, sorted(predictions[column].dropna().unique()))
        if selected:
            predictions = predictions[predictions[column].isin(selected)]
st.dataframe(predictions.sort_values("erro_absoluto_usd_m", ascending=False), use_container_width=True)

st.subheader("Maiores erros absolutos")
st.dataframe(predictions.nlargest(20, "erro_absoluto_usd_m"), use_container_width=True)

with st.expander("Métricas segmentadas e diagnóstico"):
    for filename, label in (
        ("metricas_por_porte.csv", "Por porte"),
        ("metricas_por_setor.csv", "Por setor"),
        ("vif.csv", "VIF"),
        ("coeficientes.csv", "Coeficientes"),
        ("diagnostico_particoes.csv", "Dispersão do alvo por partição"),
    ):
        path = reports_dir / filename
        if path.exists():
            st.caption(label)
            st.dataframe(pd.read_csv(path), use_container_width=True)

for filename in (
    "previsto_vs_real.png",
    "histograma_residuos.png",
    "residuos_vs_previsto.png",
    "coeficientes.png",
):
    path = reports_dir / filename
    if path.exists():
        st.image(str(path), use_container_width=True)
