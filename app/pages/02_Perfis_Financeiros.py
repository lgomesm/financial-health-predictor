import pandas as pd
import plotly.express as px
import streamlit as st
from config.settings import settings

st.title("Perfis financeiros")

result_path = settings.processed_data_dir / "empresas_com_clusters_compact.csv"

if not result_path.exists():
    st.warning("Resultado ainda não encontrado")
    st.stop()

data = pd.read_csv(result_path)
counts = (data.groupby(["cluster", "perfil_financeiro"]).size().reset_index(name="empresas"))

st.plotly_chart(px.bar(counts, x="perfil_financeiro", y="empresas", color="cluster", 
        title="Empresas por perfil"), use_container_width=True)

st.plotly_chart(px.scatter(data, x="debt_to_equity", y="liquidez_corrente", color="perfil_financeiro",
        hover_name="nome_empresa_ficticio", title="Endividamento versus liquidez"), 
    use_container_width=True)

summary_columns = ["receita_liquida_usd_m", "crescimento_receita_pct", "margem_liquida_pct",
    "debt_to_equity", "liquidez_corrente", "indice_solvencia"]

st.subheader("Médias que diferenciam os perfis")
st.dataframe(data.groupby("perfil_financeiro")[summary_columns].mean(), use_container_width=True)