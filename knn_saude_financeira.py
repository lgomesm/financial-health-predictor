"""
KNN - Classificacao de Saude Financeira de Empresas
===================================================

O KNN esta isolado na funcao `aplicar_knn(df, ...)`, que aceita QUALQUER
DataFrame: a base inteira (etapa 1) ou um unico cluster gerado pelo K-means
(etapa 3). Assim o mesmo KNN pode ser reaplicado em cada agrupamento e os
resultados comparados com o KNN geral.

Uso direto (etapa 1 - KNN na base completa):
    python knn_saude_financeira.py

Uso como biblioteca (etapa 3 - KNN por cluster), no script do K-means:
    from knn_saude_financeira import aplicar_knn, carregar_dados

    df = carregar_dados()
    # ... colega gera a coluna 'cluster' com o K-means ...
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        print(f"\\n===== Cluster {c} ({len(sub)} empresas) =====")
        aplicar_knn(sub)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV = "dataset_saude_financeira_8000_empresas.csv"

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

# Grade de hiperparametros testada na busca do melhor KNN.
GRID_KNN = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}


def carregar_dados(caminho=CSV):
    """Le o CSV e retorna o DataFrame completo."""
    df = pd.read_csv(caminho, encoding="utf-8")
    print(f"Base carregada: {df.shape[0]} empresas, {df.shape[1]} colunas")
    return df


def separar_features_e_alvo(df, target=TARGET):
    """Retorna (X, y) prontos: remove ids/vazamentos e faz one-hot."""
    y = df[target]

    # Descarta rotulo, ids, vazamentos e QUALQUER coluna 'target_*'.
    # 'cluster' tambem sai, caso o df ja tenha vindo rotulado pelo K-means.
    descartar = set(ID_COLS) | set(LEAKAGE_COLS) | {"cluster"}
    descartar |= {c for c in df.columns if c.startswith("target_")}

    X = df.drop(columns=[c for c in descartar if c in df.columns])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # One-hot nas categoricas (setor, pais, porte, etc.).
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X, y


def aplicar_knn(df, target=TARGET, test_size=0.25, random_state=42, verbose=True):
    """
    Aplica o pipeline completo de KNN num DataFrame qualquer.

    Serve tanto para a base inteira quanto para um unico cluster do K-means.
    Retorna um dict com os resultados (ou None se o subconjunto for pequeno
    ou tiver uma unica classe, casos comuns em clusters).
    """
    y = df[target]
    n_classes = y.nunique()

    # Guardas para clusters pequenos / degenerados.
    if len(df) < 20 or n_classes < 2:
        if verbose:
            motivo = "uma unica classe" if n_classes < 2 else "poucas amostras"
            print(f"  [ignorado] KNN nao aplicavel ({motivo}, n={len(df)}).")
        return None

    X, y = separar_features_e_alvo(df, target)

    if verbose:
        print(f"Distribuicao do alvo ({target}):")
        print(y.value_counts().to_string())
        print(f"Features apos one-hot: {X.shape[1]}")

    # Estratifica so se cada classe tiver ao menos 2 exemplos.
    menor_classe = y.value_counts().min()
    estratificar = y if menor_classe >= 2 else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=estratificar
    )

    # KNN e baseado em distancia -> ESCALONAR e obrigatorio.
    # Pipeline garante que o scaler aprende so no treino (sem vazamento).
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    # k nao pode ser maior que o n. de amostras de treino; ajusta a grade.
    max_k = max(1, min(len(X_tr) - 1, 21))
    grid = dict(GRID_KNN)
    grid["knn__n_neighbors"] = [k for k in GRID_KNN["knn__n_neighbors"] if k <= max_k] or [1]

    # Numero de folds da CV limitado pela menor classe do treino.
    n_folds = int(min(5, y_tr.value_counts().min()))
    if n_folds < 2:
        # Poucos exemplos por classe: treina um KNN simples, sem busca.
        modelo = pipe.set_params(knn__n_neighbors=min(5, max_k)).fit(X_tr, y_tr)
        melhores = {"knn__n_neighbors": min(5, max_k)}
        score_cv = float("nan")
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        busca = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        busca.fit(X_tr, y_tr)
        modelo = busca.best_estimator_
        melhores = busca.best_params_
        score_cv = busca.best_score_

    y_pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    labels = sorted(y.unique())

    if verbose:
        print(f"\nMelhores parametros: {melhores}")
        if score_cv == score_cv:  # nao e NaN
            print(f"F1-macro (validacao cruzada): {score_cv:.3f}")
        print(f"Acuracia (teste): {acc:.3f}")
        print("\n=== Relatorio no conjunto de teste ===")
        print(classification_report(y_te, y_pred, zero_division=0))
        print("Matriz de confusao (linhas=real, colunas=previsto):")
        print(pd.DataFrame(
            confusion_matrix(y_te, y_pred, labels=labels),
            index=labels, columns=labels,
        ))

    return {
        "modelo": modelo,
        "melhores_parametros": melhores,
        "f1_macro_cv": score_cv,
        "acuracia_teste": acc,
        "relatorio": classification_report(y_te, y_pred, zero_division=0, output_dict=True),
        "matriz_confusao": confusion_matrix(y_te, y_pred, labels=labels),
        "labels": labels,
        "n_amostras": len(df),
    }


def main():
    df = carregar_dados()
    print(f"Nulos na base: {df.isna().sum().sum()}\n")
    aplicar_knn(df)


if __name__ == "__main__":
    main()
