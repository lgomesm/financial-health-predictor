from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from data.dataset_loader import load_dataset
from domain.dataset_schema import CATEGORICAL_COLUMNS, FINANCIAL_COLUMNS
from features.feature_catalog import SYNTHETIC_RISK_COLUMNS
from features.financial_ratios import add_financial_ratios
from features.preprocessing import build_preprocessor

#Estou usando essa seed fixa pro K-Means ser reproduzível. Penso que, para os gráficos e  relatórios que vamos apresentar no artigo,
#seria interessante utilizar os resultados obtidos a partir dessa seed
RANDOM_STATE = 42
DEFAULT_CLUSTER_COUNT = 5
ARTIFACT_VERSION = "2.0.0"

ExperimentName = Literal["baseline", "compact"]

#Calculo as razões financeiras durante a própria execução do pipeline, a partir do nosso CSV de entrada
DERIVED_RATIO_COLUMNS = (
    "margem_bruta_calculada",
    "margem_ebitda_calculada",
    "margem_liquida_calculada",
    "divida_sobre_ativos",
    "caixa_sobre_passivos_circulantes",
    "fluxo_operacional_sobre_receita",
    "margem_fluxo_caixa_livre_calculada",
)

#Usei o baseline como ponto de partida (a primeira versão do modelo). Comparei a outra versão do pipeline (compact) para avaliar melhorias
#de desempenho no k-means e chegar no melhor número de clusters e variações durante o treinamento
BASELINE_NUMERIC_FEATURES = (
    tuple(
        column for column in FINANCIAL_COLUMNS if column not in SYNTHETIC_RISK_COLUMNS
    )
    + DERIVED_RATIO_COLUMNS
)

#Escolhi o modelo compacto pra representar perfis financeiros apenas, sem separação por setores. Estou evitando aqui valores monetários 
#absolutos porque, a partir dos testes, percebi que estavam formando clusters de tamanho em vez de comportamento financeiro
COMPACT_NUMERIC_FEATURES = (
    "crescimento_receita_pct",
    "margem_ebitda_pct",
    "margem_liquida_pct",
    "divida_sobre_ativos",
    "cobertura_juros",
    "liquidez_corrente",
    "margem_fluxo_caixa_livre_calculada",
    "giro_ativo",
)

#Como a média que o k-means forma é mto sensível a valores extremos, varias observaçoes mto altas ou mto baixas estavam deslocando o centro
#e prejudicando a qualidade dos agrupamentos. Por isso, estou limitando as variáveis aos percentis de 1 e 99 (p01 e p99), que calculo só com 
#os dados de treino
CLIPPED_COMPACT_FEATURES = (
    "crescimento_receita_pct",
    "margem_liquida_pct",
    "divida_sobre_ativos",
    "cobertura_juros",
    "margem_fluxo_caixa_livre_calculada",
)

#Variáveis mais usadas pra gerar relatórios e facilitar a interpretação dos clusters encontrados pelo K-Means e criar insumo para o artigo
#final. Não uso tds elas no cálculo da distância entre as observações durante o agrupamento. Por ex, setor e porte são exibidas para termos insumo 
#e podermos explicar as características de cada cluster, mas nãoi influenciam a forma como os grupos são criados
SUMMARY_FEATURES = (
    "receita_liquida_usd_m",
    "crescimento_receita_pct",
    "margem_ebitda_pct",
    "margem_liquida_pct",
    "divida_sobre_ativos",
    "debt_to_equity",
    "liquidez_corrente",
    "indice_solvencia",
    "fluxo_caixa_livre_usd_m",
    "giro_ativo",
)

class QuantileClipper(BaseEstimator, TransformerMixin):
    """Reduzindo a influência dos valores extremos aprendendo, durante o treinamento, os valores correspondentes aos percentis inferior e 
    superior (p01 e p99). Dps, substituo qualquer valor fora desse intervalo pelo limite mais próximo

    E quanto aos limites... aprendo apenas com os dados de treino e reutilizo em validação, teste e produção
    """

    def __init__(
        self, columns: tuple[str, ...], lower: float = 0.01, upper: float = 0.99
    ):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, data: pd.DataFrame, y: object = None) -> QuantileClipper:
        self.lower_bounds_ = data.loc[:, self.columns].quantile(self.lower)
        self.upper_bounds_ = data.loc[:, self.columns].quantile(self.upper)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = data.copy()
        transformed.loc[:, self.columns] = transformed.loc[:, self.columns].clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1
        )
        return transformed

class FinancialRatiosTransformer(BaseEstimator, TransformerMixin):
    """Calculo os indicadores financeiros usados pelo modelo. Para isso, pego os dados brutos do dataset e converto em indicadores financeiros 
    que vão ser usados nas próx etapas do fluxo

    Obs: estou executando o transformer antes do clipping porque os limites pra tratamento dos outliers precisam ser aprendidos sobre os 
    indicadores que já calculei
    """
    def fit(self, data: pd.DataFrame, y: object = None) -> FinancialRatiosTransformer:
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = data.copy()
        for column in FINANCIAL_COLUMNS:
            if column in transformed:
                transformed[column] = pd.to_numeric(
                    transformed[column], errors="coerce"
                )
        transformed = add_financial_ratios(transformed)
        transformed.loc[:, BASELINE_NUMERIC_FEATURES] = transformed.loc[
            :, BASELINE_NUMERIC_FEATURES
        ].replace([np.inf, -np.inf], np.nan)
        return transformed

@dataclass(frozen=True, slots=True)
class ClusteringMetrics:
    n_clusters: int
    inertia: float
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_score: float
    smallest_cluster_size: int
    largest_cluster_size: int
    mean_distance_to_assigned_centroid: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

@dataclass(slots=True)
class ClusteringResult:
    """Agrupando os artefatos de uma execução sem esconder o modelo treinado (Obs: Novamente, usei para avaliar o melhor equilibrio de 
    métricas para o k-means)
    """
    experiment: ExperimentName
    pipeline: Pipeline
    clustered_data: pd.DataFrame
    metrics_by_partition: pd.DataFrame
    elbow_metrics: pd.DataFrame
    cluster_summary: pd.DataFrame
    numeric_centroids: pd.DataFrame
    pca_coordinates: np.ndarray
    pca_model: PCA
    distances_to_centroid: np.ndarray
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

def load_and_prepare_dataset(path: Path) -> pd.DataFrame:
    return load_dataset(path).copy()

def run_kmeans(
    dataset: pd.DataFrame,
    *,
    experiment: ExperimentName = "compact",
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
    random_state: int = RANDOM_STATE,
    elbow_range: range = range(2, 11),
) -> ClusteringResult:
    """
    Treino o K-Means usando só os dados de treinamento e, dps,
    uso o modelo pra classificar as outras observações (sem fazer um treinamento novo)

    Quando chamo o fit, td fluxo já é ajustado com os dados de treino (incluindo o cálculo dos limites do clipping por percentis, o aprendizado
    das estatísticas do pré-processamento e os centróides do k-means

    O n_clusters foi algo que defini de primeiro momento. Fui testando diferentes valores de k são testados e comparando pra
    identificar qual gerava o melhor agrupamento

    Por fim, uso o n_init=20 pra que o algoritmo execute o treinamento 20 vezes, cada uma com uma inicialização
    diferente dos centróides. No final, mantenho só a solução com menos inércia (soma das distâncias entre as observações e os centróides)
    """
    numeric_features, categorical_features = _features_for(experiment)
    _validate_input(dataset, categorical_features, n_clusters)

    train = dataset.loc[dataset["split_ml"] == "treino"].copy()
    pipeline = _build_pipeline(
        experiment,
        numeric_features,
        categorical_features,
        n_clusters,
        random_state,
    )
    pipeline.fit(train)

    #Predict percorre tds as etapas anteriores e atribui o centróide de menor distância euclidiana (não "aprende" nada nessa chamada)
    clustered = pipeline.named_steps["financial_ratios"].transform(dataset)
    clustered["cluster"] = pipeline.predict(dataset)
    clustered["perfil_financeiro"] = clustered["cluster"].map(
        lambda cluster: f"Cluster {cluster}"
    )

    distances = _assigned_distances(pipeline, dataset, clustered["cluster"].to_numpy())
    clustered["distancia_ao_centroide"] = distances
    metrics = _metrics_by_partition(clustered, pipeline)

    #a matriz é a mesma que chegou ao k-means no treino, só que padronizada
    transformed_train = pipeline[:-1].transform(train)
    elbow = _calculate_elbow_metrics(transformed_train, elbow_range, random_state)
    centroids = _numeric_centroids(pipeline, numeric_features)
    summary = _build_cluster_summary(clustered)

    pca_model = PCA(n_components=2, random_state=random_state)
    pca_model.fit(transformed_train)
    pca_coordinates = pca_model.transform(pipeline[:-1].transform(dataset))

    return ClusteringResult(
        experiment=experiment,
        pipeline=pipeline,
        clustered_data=clustered,
        metrics_by_partition=metrics,
        elbow_metrics=elbow,
        cluster_summary=summary,
        numeric_centroids=centroids,
        pca_coordinates=pca_coordinates,
        pca_model=pca_model,
        distances_to_centroid=distances,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def predict_new_companies(pipeline: Pipeline, dataset: pd.DataFrame) -> np.ndarray:
    """Atribui os clusters a empresas novas usando os parâmetros já aprendidos"""
    return pipeline.predict(dataset)


def export_results(
    result: ClusteringResult,
    output_dir: Path,
    reports_dir: Path,
    models_dir: Path,
    *,
    dataset_path: Path | None = None,
) -> None:
    """Exporta os resultados legíveis. Utilizei para as comparações de performance dos clusters encontrados durante o treinamento"""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    result.clustered_data.to_csv(
        output_dir / f"empresas_com_clusters_{result.experiment}.csv", index=False
    )
    result.metrics_by_partition.to_csv(
        reports_dir / "metricas_por_particao.csv", index=False
    )
    result.elbow_metrics.to_csv(reports_dir / "metodo_cotovelo.csv", index=False)
    result.cluster_summary.to_csv(reports_dir / "resumo_clusters.csv", index=False)
    result.numeric_centroids.to_csv(
        reports_dir / "centroides_numericos.csv", index=False
    )
    _export_cluster_diagnostics(result.clustered_data, reports_dir)

    artifact_path = (
        models_dir / f"kmeans_{result.experiment}_v{ARTIFACT_VERSION}.joblib"
    )
    pca_path = models_dir / f"kmeans_{result.experiment}_pca_v{ARTIFACT_VERSION}.joblib"
    joblib.dump(result.pipeline, artifact_path)
    joblib.dump(result.pca_model, pca_path)

    kmeans_model = result.pipeline.named_steps["kmeans"]
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "experiment": result.experiment,
        "artifact_file": artifact_path.name,
        "pca_artifact_file": pca_path.name,
        "dataset_name": dataset_path.name if dataset_path else None,
        "dataset_hash_sha256": _file_hash(dataset_path) if dataset_path else None,
        "n_clusters": int(kmeans_model.n_clusters),
        #leio do modelo
        "random_state": int(kmeans_model.random_state),
        "numeric_features": list(result.numeric_features),
        "categorical_features": list(result.categorical_features),
        #pending_human_validation porque, ainda que os clusters sejam encontrados, a interpretação feita será do nosso lado
        "profile_mapping_status": "pending_human_validation",
        "cluster_labels": {
            str(cluster): f"Cluster {cluster}"
            for cluster in range(int(kmeans_model.n_clusters))
        },
        "fit_partition": "treino",
        "partition_rows": result.clustered_data["split_ml"].value_counts().to_dict(),
        "metrics_by_partition": _json_safe(
            result.metrics_by_partition.set_index("particao").to_dict(orient="index")
        ),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "outlier_clipping": _clipping_metadata(result.pipeline),
    }
    metadata_path = models_dir / f"kmeans_{result.experiment}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _features_for(
    experiment: ExperimentName,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Define, explicitamente, quais colunas entram em cada experimento."""
    if experiment == "compact":
        #sem categorias, porque evita que varios one-hot encodings ficassem formando clusters
        #por setor ou país, por exemplo 
        return COMPACT_NUMERIC_FEATURES, ()
    if experiment == "baseline":
        return BASELINE_NUMERIC_FEATURES, CATEGORICAL_COLUMNS
    raise ValueError(f"Unknown experiment: {experiment}")


def _build_pipeline(
    experiment: ExperimentName,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    n_clusters: int,
    random_state: int,
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("financial_ratios", FinancialRatiosTransformer())
    ]
    if experiment == "compact":
        steps.append(("outlier_clipper", QuantileClipper(CLIPPED_COMPACT_FEATURES)))
    steps.extend(
        [
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=20,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def _validate_input(
    dataset: pd.DataFrame,
    categorical: tuple[str, ...],
    n_clusters: int,
) -> None:
    required = set(FINANCIAL_COLUMNS + categorical + ("split_ml",))
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing clustering features: {', '.join(missing)}"
        )

    training_rows = int((dataset["split_ml"] == "treino").sum())
    if not 2 <= n_clusters < training_rows:
        raise ValueError("n_clusters must be between 2 and the number of training rows")

    expected_splits = {"treino", "validacao", "teste"}
    missing_splits = sorted(expected_splits - set(dataset["split_ml"]))
    if missing_splits:
        raise ValueError(
            f"Dataset is missing split_ml values: {', '.join(missing_splits)}"
        )


def _metrics_by_partition(clustered: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Mede separação e generalização sem permitir que validação/teste re-treinem."""
    rows: list[dict[str, float | int | str]] = []
    for partition in ("treino", "validacao", "teste"):
        subset = clustered.loc[clustered["split_ml"] == partition]
        transformed = pipeline[:-1].transform(subset)
        labels = subset["cluster"].to_numpy()
        #inércia é objetivo de ajuste do k-means, por isso só defino pra treino
        inertia = (
            float(pipeline.named_steps["kmeans"].inertia_)
            if partition == "treino"
            else float("nan")
        )
        metrics = _calculate_metrics(transformed, labels, inertia).to_dict()
        metrics["particao"] = partition
        metrics["mean_distance_to_assigned_centroid"] = float(
            subset["distancia_ao_centroide"].mean()
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _calculate_metrics(
    transformed: np.ndarray, labels: np.ndarray, inertia: float
) -> ClusteringMetrics:
    """Calculo de métricas

    * Silhouette: próximo de 1 indica os pontos próximos do próprio grupo e distantes
      dos demais (valores baixos podem acontecer, e aconteceram em alguns casos, devido a sobreposição)
    * Davies-Bouldin: menor é melhor (compara dispersão interna e separação)
    * Calinski-Harabasz: maior é melhor (razão entre separação e dispersão)

    Uso tamanho mínimo/máximo visando evitar o risco de escolher um k com boa métrica mas grupos residuais, e a distância média 
    indica se novas empresas ficam longe dos centróides que foram aprendidos
    """
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        raise ValueError("At least two predicted clusters are required for evaluation")
    return ClusteringMetrics(
        n_clusters=int(len(counts)),
        inertia=float(inertia),
        silhouette_score=float(silhouette_score(transformed, labels)),
        davies_bouldin_index=float(davies_bouldin_score(transformed, labels)),
        calinski_harabasz_score=float(calinski_harabasz_score(transformed, labels)),
        smallest_cluster_size=int(counts.min()),
        largest_cluster_size=int(counts.max()),
        mean_distance_to_assigned_centroid=float("nan"),
    )


def _assigned_distances(
    pipeline: Pipeline, dataset: pd.DataFrame, labels: np.ndarray
) -> np.ndarray:
    """Retorna a distância euclidiana de cada empresa ao centróide atribuído."""
    distances_to_all_centroids = pipeline.transform(dataset)
    return distances_to_all_centroids[np.arange(len(dataset)), labels]


def _calculate_elbow_metrics(
    transformed: np.ndarray, elbow_range: range, random_state: int
) -> pd.DataFrame:
    """Compara inércia e métricas para vários k

    Lembrando que a inércia sempre cai quando k aumenta, porque mais centróides aproximam melhor os
    pontos"""
    rows: list[dict[str, float | int]] = []
    for count in elbow_range:
        if count >= len(transformed):
            continue
        model = KMeans(n_clusters=count, random_state=random_state, n_init=20).fit(
            transformed
        )
        rows.append(
            _calculate_metrics(transformed, model.labels_, model.inertia_).to_dict()
        )
    return pd.DataFrame(rows)


def _numeric_centroids(
    pipeline: Pipeline, numeric_features: tuple[str, ...]
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    centers = pipeline.named_steps["kmeans"].cluster_centers_[
        :, : len(numeric_features)
    ]
    centroids = pd.DataFrame(
        scaler.inverse_transform(centers), columns=numeric_features
    )
    centroids.insert(0, "cluster", range(len(centroids)))
    return centroids


def _build_cluster_summary(clustered: pd.DataFrame) -> pd.DataFrame:
    """Produz médias, medianas, extremos e quartis para interpretação"""
    grouped = clustered.groupby("cluster")
    statistics = grouped[list(SUMMARY_FEATURES)].agg(["mean", "median", "min", "max"])
    statistics.columns = [
        f"{feature}_{statistic}" for feature, statistic in statistics.columns
    ]
    summary = statistics.reset_index()
    summary.insert(
        1, "perfil_financeiro", summary["cluster"].map(lambda value: f"Cluster {value}")
    )
    summary = summary.merge(
        grouped.size().rename("quantidade_empresas").reset_index(),
        on="cluster",
        how="left",
    )

    #valores extremos estavam deslocando a média. por conta disso, uso q1, mediana e q3 pra descrever a distribuição dos dados
    quartiles = grouped[list(SUMMARY_FEATURES)].quantile([0.25, 0.75]).unstack(level=-1)
    quartiles.columns = [
        f"{feature}_{'q1' if quantile == 0.25 else 'q3'}"
        for feature, quantile in quartiles.columns
    ]
    return summary.merge(quartiles.reset_index(), on="cluster", how="left")


def _export_cluster_diagnostics(clustered: pd.DataFrame, reports_dir: Path) -> None:
    clustered.groupby(["cluster", "split_ml"]).size().reset_index(
        name="empresas"
    ).to_csv(reports_dir / "empresas_por_cluster_e_particao.csv", index=False)

    compositions: list[pd.DataFrame] = []
    for column in ("setor_economico", "porte_empresa", "estagio_maturidade"):
        composition = (
            clustered.groupby(["cluster", column]).size().reset_index(name="empresas")
        )
        composition.insert(1, "variavel", column)
        compositions.append(composition.rename(columns={column: "categoria"}))
    pd.concat(compositions, ignore_index=True).to_csv(
        reports_dir / "composicao_categorica_por_cluster.csv", index=False
    )

    columns = [
        "cluster",
        "id_empresa",
        "nome_empresa_ficticio",
        "split_ml",
        "distancia_ao_centroide",
    ]

    clustered.sort_values("distancia_ao_centroide").groupby("cluster").head(5).loc[
        :, columns
    ].to_csv(reports_dir / "empresas_tipicas.csv", index=False)
    clustered.sort_values("distancia_ao_centroide", ascending=False).groupby(
        "cluster"
    ).head(5).loc[:, columns].to_csv(
        reports_dir / "empresas_de_fronteira.csv", index=False
    )


def _clipping_metadata(pipeline: Pipeline) -> dict[str, dict[str, float]] | None:
    clipper = pipeline.named_steps.get("outlier_clipper")
    if clipper is None:
        return None
    return {
        column: {
            "p01": float(clipper.lower_bounds_[column]),
            "p99": float(clipper.upper_bounds_[column]),
        }
        for column in clipper.columns
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def evaluate_cluster_counts(
    dataset: pd.DataFrame,
    *,
    experiment: ExperimentName = "compact",
    cluster_counts: range = range(2, 9),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for count in cluster_counts:
        result = run_kmeans(
            dataset,
            experiment=experiment,
            n_clusters=count,
            random_state=random_state,
            elbow_range=range(2, 3),
        )
        rows.append(result.metrics_by_partition.assign(n_clusters_requested=count))
    return pd.concat(rows, ignore_index=True)


def evaluate_seed_stability(
    dataset: pd.DataFrame,
    *,
    experiment: ExperimentName = "compact",
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
    seeds: tuple[int, ...] = (42, 7, 21, 84, 123),
) -> pd.DataFrame:
    reference: np.ndarray | None = None
    rows: list[dict[str, float | int]] = []
    for seed in seeds:
        result = run_kmeans(
            dataset,
            experiment=experiment,
            n_clusters=n_clusters,
            random_state=seed,
            elbow_range=range(2, 3),
        )
        labels = result.clustered_data["cluster"].to_numpy()
        rows.append(
            {
                "seed": seed,
                "adjusted_rand_index_vs_seed_42": 1.0
                if reference is None
                else float(adjusted_rand_score(reference, labels)),
            }
        )
        if reference is None:
            reference = labels
    return pd.DataFrame(rows)
