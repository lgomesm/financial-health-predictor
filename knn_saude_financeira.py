"""
KNN - Classificacao de Saude Financeira de Empresas
===================================================

Script inicial (etapa 1 do trabalho): aplica KNN na base completa para
classificar as empresas por faixa de saude financeira.

Etapas seguintes do projeto (nao implementadas aqui):
  2. Remover a coluna de classe e aplicar K-means para clusterizar.
  3. Aplicar KNN dentro de cada cluster e comparar com o KNN geral.

Uso:
    python knn_saude_financeira.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

CSV = "dataset_saude_financeira_8000_empresas(2).csv"

# Rotulo que queremos prever: a faixa de saude financeira.
TARGET = "target_faixa_saude_financeira"

# --------------------------------------------------------------------------
# COLUNAS QUE NAO PODEM ENTRAR COMO FEATURE
# --------------------------------------------------------------------------
# 1) Identificadores / metadados: nao sao informacao financeira.
ID_COLS = [
    "id_empresa", "nome_empresa_ficticio", "ano_referencia", "split_ml",
]

# 2) VAZAMENTO DE DADOS (data leakage):
#    - Todas as outras colunas 'target_*' sao rotulos/derivados do futuro.
#    - 'indice_saude_financeira_0_100' e 'score_financeiro_geral_0_100'
#      determinam DIRETAMENTE a faixa (as faixas sao apenas fatias desse
#      indice: Critica 1-38, Atencao 38-60, Boa 60-78, Excelente 78-100).
#      Se voce deixar essas colunas, o KNN acerta ~100% de forma trivial e
#      o modelo nao aprende nada util. Por isso removemos.
LEAKAGE_COLS = [
    "indice_saude_financeira_0_100",   # define a faixa quase 1:1
    "score_financeiro_geral_0_100",    # score composto do rotulo
]


def carregar_dados():
    df = pd.read_csv(CSV, encoding="utf-8")
    print(f"Base carregada: {df.shape[0]} empresas, {df.shape[1]} colunas")
    print(f"\nDistribuicao do alvo ({TARGET}):")
    print(df[TARGET].value_counts())
    print(f"\nNulos na base: {df.isna().sum().sum()}")
    return df


def separar_features_e_alvo(df):
    y = df[TARGET]

    # Descarta rotulo, ids e vazamentos. Todas as colunas 'target_*' saem.
    descartar = set(ID_COLS) | set(LEAKAGE_COLS)
    descartar |= {c for c in df.columns if c.startswith("target_")}

    X = df.drop(columns=[c for c in descartar if c in df.columns])

    # Separa numericas x categoricas.
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print(f"\nFeatures usadas: {X.shape[1]} "
          f"({len(num_cols)} numericas, {len(cat_cols)} categoricas)")
    print(f"Categoricas: {cat_cols}")

    # One-hot nas categoricas (setor, pais, porte, etc.).
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"Apos one-hot encoding: {X.shape[1]} features")
    return X, y


def treinar_knn(X, y):
    # Estratifica para manter a proporcao das classes (base desbalanceada).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # KNN e baseado em distancia -> ESCALONAR e obrigatorio.
    # Pipeline garante que o scaler aprende so no treino (sem vazamento).
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    # Busca o melhor k e o esquema de pesos.
    grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busca = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    busca.fit(X_tr, y_tr)

    print(f"\nMelhores parametros: {busca.best_params_}")
    print(f"F1-macro (validacao cruzada): {busca.best_score_:.3f}")

    # Avaliacao final no conjunto de teste.
    y_pred = busca.predict(X_te)
    print("\n=== Relatorio no conjunto de teste ===")
    print(classification_report(y_te, y_pred))
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    labels = sorted(y.unique())
    print(pd.DataFrame(
        confusion_matrix(y_te, y_pred, labels=labels),
        index=labels, columns=labels,
    ))
    return busca.best_estimator_


def main():
    df = carregar_dados()
    X, y = separar_features_e_alvo(df)
    treinar_knn(X, y)


if __name__ == "__main__":
    main()
