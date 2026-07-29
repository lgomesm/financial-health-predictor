import pandas as pd
import streamlit as st

from config.settings import settings

st.title("Visão geral do dataset")

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(settings.raw_data_dir / "dataset_saude_financeira_8000_empresas.csv")

try:
    data = load_data()
    st.metric("Empresas", len(data))
    st.dataframe(data.head(), use_container_width=True)
except Exception as error:
    st.error(f"Não foi possível carregar o dataset: {error}")