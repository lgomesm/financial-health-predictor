import json

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
counts = (
    data.groupby(["cluster", "perfil_financeiro"]).size().reset_index(name="empresas")
)

st.plotly_chart(
    px.bar(
        counts,
        x="perfil_financeiro",
        y="empresas",
        color="cluster",
        title="Empresas por perfil",
    ),
    use_container_width=True,
)

st.plotly_chart(
    px.scatter(
        data,
        x="debt_to_equity",
        y="liquidez_corrente",
        color="perfil_financeiro",
        hover_name="nome_empresa_ficticio",
        title="Endividamento versus liquidez",
    ),
    use_container_width=True,
)

summary_columns = [
    "receita_liquida_usd_m",
    "crescimento_receita_pct",
    "margem_liquida_pct",
    "debt_to_equity",
    "liquidez_corrente",
    "indice_solvencia",
]

st.subheader("Médias que diferenciam os perfis")
st.dataframe(
    data.groupby("perfil_financeiro")[summary_columns].mean(), use_container_width=True
)

knn_reports_dir = settings.reports_dir / "kmeans" / "compact" / "validacao_knn"
knn_summary_path = knn_reports_dir / "resumo_validacao_knn.json"
knn_demo_path = knn_reports_dir / "demonstracoes_knn.json"

if knn_summary_path.exists() and knn_demo_path.exists():
    knn_summary = json.loads(knn_summary_path.read_text(encoding="utf-8"))
    knn_demo = json.loads(knn_demo_path.read_text(encoding="utf-8"))
    st.divider()
    st.header("Validação local dos clusters com KNN")
    st.caption(
        "O KNN verificou se empresas vizinhas conseguem reproduzir os clusters atribuídos pelo K-Means"
    )
    metrics_columns = st.columns(3)
    metrics_columns[0].metric(
        "Concordância no teste", f"{knn_summary['test']['accuracy']:.1%}"
    )
    metrics_columns[1].metric(
        "Macro F1 no teste", f"{knn_summary['test']['macro_f1']:.3f}"
    )
    metrics_columns[2].metric(
        "Balanced accuracy", f"{knn_summary['test']['balanced_accuracy']:.3f}"
    )
    st.write("Parâmetros KNN selecionados:", knn_summary["best_parameters"])
    for observation in knn_summary["interpretations"]["test"]:
        st.write(f"- {observation}")

    confusion_path = knn_reports_dir / "matriz_confusao_test.csv"
    if confusion_path.exists():
        st.subheader("Matriz de confusão — K-Means x KNN no teste")
        st.dataframe(pd.read_csv(confusion_path, index_col=0), use_container_width=True)

    st.subheader("Empresas típicas")
    for company in knn_demo["typical_companies"]:
        votes = company["vizinhos"]["vote_distribution"]
        st.write(
            f"**{company['nome_empresa_ficticio']}** — K-Means: Cluster "
            f"{company['cluster_kmeans']}; KNN: Cluster {company['cluster_knn']}; "
            f"vizinhos: {votes}."
        )

    boundary = knn_demo["boundary_company"]
    st.subheader("Empresa de fronteira")
    st.write(
        f"**{boundary['nome_empresa_ficticio']}** — K-Means: Cluster "
        f"{boundary['cluster_kmeans']}; KNN: Cluster {boundary['cluster_knn']}; "
        f"vizinhos: {boundary['vizinhos']['vote_distribution']}."
    )
else:
    st.info("Execute validar_clusters_com_knn.py para exibir a validação local do KNN.")
