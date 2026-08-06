import argparse
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from config.settings import settings
from services.kmeans_clustering_service import (DEFAULT_CLUSTER_COUNT, SUMMARY_FEATURES, 
    ClusteringResult, evaluate_cluster_counts, evaluate_seed_stability, export_results, 
    load_and_prepare_dataset, run_kmeans)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run financial K-Means experiments")
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--experiment", choices=("baseline", "compact", "both"), default="both")
    parser.add_argument("--input", type=Path, 
        default=settings.raw_data_dir / "dataset_saude_financeira_8000_empresas.csv",
    )
    parser.add_argument("--skip-robustness-checks", action="store_true")

    return parser.parse_args()

def save_figure(figure: Figure, output_path: Path, dpi: int = 150) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)

    plt.close(figure)

def create_scatter_plot(x_values: Any, y_values: Any, clusters: pd.Series, title: str, x_label: str, y_label: str) -> Figure:
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(x_values, y_values, c=clusters, cmap="tab10", s=10, alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    return figure


def create_pca_clusters_with_centroids_plot(result: ClusteringResult) -> Figure:
    """Cria a projeção PCA usada para interpretar visualmente os clusters.

    A PCA já foi ajustada dentro de ``run_kmeans`` exclusivamente na partição de
    treino. Portanto, esta função não chama ``fit``: ela apenas reutiliza as
    coordenadas e o modelo aprendidos para desenhar todas as empresas e os
    centroides no mesmo espaço bidimensional.
    """
    figure, axis = plt.subplots(figsize=(10, 7))
    clusters = result.clustered_data["cluster"].to_numpy()
    color_map = plt.get_cmap("tab10")

    # Desenhar cada grupo separadamente permite identificá-lo com clareza na legenda.
    for cluster in sorted(set(clusters)):
        cluster_mask = clusters == cluster
        axis.scatter(
            result.pca_coordinates[cluster_mask, 0],
            result.pca_coordinates[cluster_mask, 1],
            color=color_map(int(cluster)),
            s=12,
            alpha=0.45,
            label=f"Cluster {cluster}",
        )

    # Os centros do K-Means estão no espaço padronizado de oito indicadores,
    # exatamente o espaço usado no ajuste da PCA; por isso podem ser projetados
    # pelo mesmo objeto PCA sem nenhum novo treinamento.
    kmeans = result.pipeline.named_steps["kmeans"]
    centroids_pca = result.pca_model.transform(kmeans.cluster_centers_)
    axis.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=220,
        color="white",
        edgecolors="black",
        linewidths=1.2,
        label="Centroide",
        zorder=3,
    )

    for cluster, coordinate in enumerate(centroids_pca):
        axis.annotate(
            f"Cluster {cluster}",
            xy=coordinate,
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    variance = result.pca_model.explained_variance_ratio_ * 100
    axis.set_xlabel(f"Componente principal 1 ({variance[0]:.1f}% da variância)")
    axis.set_ylabel(f"Componente principal 2 ({variance[1]:.1f}% da variância)")
    axis.set_title(
        "Projeção PCA dos perfis financeiros identificados pelo K-Means"
    )
    axis.legend(title="Perfis", loc="best")

    return figure


def save_visualizations(result: ClusteringResult, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    clustered_data = result.clustered_data
    pca_coordinates = result.pca_coordinates

    scatter_plots = [
        {
            "x_values": pca_coordinates[:, 0], "y_values": pca_coordinates[:, 1], "title": f"PCA — {result.experiment}", 
            "x_label": "Componente 1", "y_label": "Componente 2", "filename": "pca_clusters.png"
        },
        {
            "x_values": clustered_data["divida_sobre_ativos"], "y_values": clustered_data["liquidez_corrente"],
            "title": "Dívida sobre ativos versus liquidez", "x_label": "Dívida sobre ativos", "y_label": "Liquidez corrente",
            "filename": "dispersao_divida_liquidez.png",
        },
    ]

    for plot_configuration in scatter_plots:
        figure = create_scatter_plot(x_values=plot_configuration["x_values"], y_values=plot_configuration["y_values"], 
            clusters=clustered_data["cluster"], title=plot_configuration["title"], x_label=plot_configuration["x_label"],
            y_label=plot_configuration["y_label"]
        )

        output_path = reports_dir / plot_configuration["filename"]

        save_figure(figure=figure, output_path=output_path)

    # Esta versão é direcionada ao relatório: além dos pontos, mostra os
    # centroides projetados e explicita a variância preservada pela PCA.
    pca_figure = create_pca_clusters_with_centroids_plot(result)
    save_figure(
        figure=pca_figure,
        output_path=reports_dir / "pca_clusters_kmeans.png",
        dpi=300,
    )

    save_cluster_quantity_chart(data=clustered_data, output_path=reports_dir / "quantidade_por_cluster.png")
    save_cluster_means_chart(data=clustered_data, output_path=reports_dir / "medias_por_cluster.png")

def save_cluster_quantity_chart(data: pd.DataFrame, output_path: Path) -> None:
    cluster_quantity = data.groupby("cluster").size()
    figure, axis = plt.subplots(figsize=(8, 5))
    cluster_quantity.plot(kind="bar", ax=axis)
    axis.set_title("Empresas por cluster")
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Empresas")

    save_figure(figure=figure, output_path=output_path)

def save_cluster_means_chart(data: pd.DataFrame, output_path: Path) -> None:
    summary_columns = list(SUMMARY_FEATURES)
    cluster_means = (data.groupby("cluster")[summary_columns].mean())
    standardized_means = standardize_dataframe(cluster_means)
    figure, axis = plt.subplots(figsize=(12, 6))
    standardized_means.T.plot(kind="bar", ax=axis)
    axis.set_title("Médias padronizadas por cluster")
    axis.set_xlabel("Característica financeira")
    axis.set_ylabel("Média padronizada")

    save_figure(figure=figure, output_path=output_path)

def standardize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    column_means = data.mean()
    column_standard_deviations = data.std()
    safe_standard_deviations = column_standard_deviations.replace(0, 1)
    standardized_data = (data - column_means) / safe_standard_deviations

    return standardized_data

def main() -> None:
    args = parse_args()
    dataset = load_and_prepare_dataset(args.input)
    experiment_options = {
        "baseline": ["baseline"],
        "compact": ["compact"],
        "both": ["baseline", "compact"],
    }

    experiments = experiment_options[args.experiment]    
    comparison: list[pd.DataFrame] = []

    for experiment in experiments:
        result = run_kmeans(dataset, experiment=experiment, n_clusters=args.clusters)
        reports_dir = settings.reports_dir / "kmeans" / experiment
        export_results(result, settings.processed_data_dir, reports_dir, settings.models_dir, dataset_path=args.input)

        save_visualizations(result, reports_dir)

        comparison.append(result.metrics_by_partition.assign(experimento=experiment))

        print(f"\n{experiment.upper()}")
        print(result.metrics_by_partition.to_string(index=False))

    pd.concat(comparison, ignore_index=True).to_csv(settings.reports_dir / "kmeans" / "comparacao_experimentos.csv", index=False)

    if not args.skip_robustness_checks and "compact" in experiments:
        evaluate_cluster_counts(dataset, experiment="compact").to_csv(settings.reports_dir / "kmeans" / "compact" / "selecao_de_k.csv",
            index=False)

        evaluate_seed_stability(dataset, experiment="compact").to_csv(settings.reports_dir / "kmeans" / "compact" / "estabilidade_seeds.csv",
            index=False)

if __name__ == "__main__":
    main()
