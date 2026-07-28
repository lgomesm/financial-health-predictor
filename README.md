# Classificação de Saúde Financeira de Empresas — KNN

Trabalho de Sistemas Inteligentes. Base de 8.000 empresas (gerada por LLM) com
indicadores financeiros. Este módulo aplica **KNN** para classificar cada empresa
por **faixa de saúde financeira** (`Excelente`, `Boa`, `Atenção`, `Crítica`).

O KNN está isolado numa função reutilizável para poder ser aplicado tanto na
**base inteira** quanto em **cada cluster** gerado depois pelo K-means.

## Etapas do projeto

1. **KNN geral** — classifica a base completa.
2. **K-means** — remove a classe e clusteriza as empresas.
3. **KNN por cluster** — reaplica o mesmo KNN dentro de cada agrupamento e
   compara com o resultado geral.

## Pré-requisitos

- Python 3.9+
- Bibliotecas:

```powershell
python -m pip install pandas scikit-learn numpy
```

O arquivo `dataset_saude_financeira_8000_empresas.csv` deve estar na mesma pasta.

## Como rodar (etapa 1 — base inteira)

No terminal, dentro da pasta do projeto:

```powershell
python knn_saude_financeira.py
```

Se `python` não for reconhecido, use `py knn_saude_financeira.py`.

### O que é impresso

1. Nº de empresas/colunas e distribuição das classes.
2. Features usadas após o _one-hot encoding_.
3. Melhores hiperparâmetros do KNN (k, métrica, pesos) via validação cruzada.
4. Relatório por classe (_precision / recall / f1_) e acurácia geral (~75%).
5. Matriz de confusão (linhas = real, colunas = previsto).

## Como usar isoladamente (etapa 3 — por cluster)

A função `aplicar_knn(df)` aceita **qualquer** DataFrame. No script do K-means,
depois de criar a coluna `cluster`, basta importar e chamar em cada grupo:

```python
from knn_saude_financeira import carregar_dados, aplicar_knn

df = carregar_dados()

# ... aqui o K-means gera a coluna 'cluster' ...

for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    print(f"\n===== Cluster {c} ({len(sub)} empresas) =====")
    resultado = aplicar_knn(sub)          # imprime o relatório do cluster
    print("Acurácia:", resultado["acuracia_teste"])
```

`aplicar_knn` retorna um dicionário com: `modelo`, `melhores_parametros`,
`f1_macro_cv`, `acuracia_teste`, `relatorio` (dict), `matriz_confusao`,
`labels` e `n_amostras` — útil para comparar clusters entre si e contra o geral.
Passe `verbose=False` para suprimir a impressão e só coletar os números.

> A função já trata clusters pequenos ou com uma só classe (retorna `None` e
> avisa), ajusta `k` e o nº de _folds_ ao tamanho do grupo automaticamente.

## Decisões importantes do modelo

- **Alvo (`TARGET`)**: `target_faixa_saude_financeira` — já vinha pronto na base.
- **Escalonamento**: obrigatório (KNN usa distância). Feito com `StandardScaler`
  dentro de um `Pipeline`, para o scaler aprender só no treino (sem vazamento).
- **Vazamento de dados removido**: as colunas `indice_saude_financeira_0_100` e
  `score_financeiro_geral_0_100` **definem** a faixa quase 1:1. Se mantidas, o
  KNN acerta ~100% de forma artificial. Todas as outras colunas `target_*`
  também são descartadas das features.

## Onde ajustar

| O que mudar                               | Onde (em `knn_saude_financeira.py`)    |
| ----------------------------------------- | -------------------------------------- |
| Alvo (ex.: `target_classe_risco`)         | constante `TARGET`                     |
| Valores de `k` / métrica / pesos testados | dicionário `GRID_KNN`                  |
| Proporção treino/teste                    | parâmetro `test_size` de `aplicar_knn` |
| Colunas ignoradas / vazamento             | listas `ID_COLS` e `LEAKAGE_COLS`      |

## Observação

O CSV usa acentuação em _latin-1_, então nomes aparecem como `Cr�tica`/`Aten��o`
no terminal. **Não afeta o modelo** — é apenas exibição.
