# Dicionário de Dados

**Dataset:** `data/raw/dataset_saude_financeira_8000_empresas.csv`
**Projeto:** Sistema Inteligente de Diagnóstico de Saúde Financeira Corporativa
**Disciplina:** Sistemas Inteligentes — ESZI014-17 — UFABC — 2026-Q2

---

## 1. Visão geral

| Característica | Valor |
|---|---|
| Instâncias | 8.000 empresas |
| Variáveis | 79 colunas |
| Ano de referência | 2025 (constante) |
| Valores ausentes | 0 |
| Codificação | UTF-8 (com BOM) |
| Separador | `,` |
| Origem | Dados sintéticos gerados a partir do sistema idealizado |

### Composição por tipo

| Tipo | Quantidade |
|---|---|
| Numéricas contínuas (`float64`) | 64 |
| Numéricas inteiras (`int64`) | 2 |
| Categóricas / texto (`object`) | 13 |

### Partição de aprendizado (`split_ml`)

A divisão é **pré-definida no dataset** e respeitada por todos os pipelines: o ajuste
(`fit`) ocorre exclusivamente sobre `treino`, a seleção de limiares sobre `validacao`,
e `teste` é consumido uma única vez, ao final.

| Partição | Instâncias | Proporção |
|---|---|---|
| `treino` | 5.640 | 70,5% |
| `validacao` | 1.198 | 15,0% |
| `teste` | 1.162 | 14,5% |

---

## 2. Identificadores e metadados

| Variável | Tipo | Descrição | Valores |
|---|---|---|---|
| `id_empresa` | Categórica | Identificador único da empresa | 8.000 valores distintos (`EMP00001`–`EMP08000`) |
| `nome_empresa_ficticio` | Categórica | Razão social fictícia | 8.000 valores distintos |
| `ano_referencia` | Inteira | Exercício de referência | 2025 (constante — variância zero) |
| `split_ml` | Categórica | Partição de aprendizado | `treino`, `validacao`, `teste` |

> **Nota metodológica:** `id_empresa`, `nome_empresa_ficticio` e `split_ml` são excluídos
> de qualquer conjunto de atributos (ver `PROHIBITED_FEATURE_COLUMNS` em
> `src/features/feature_catalog.py`). `ano_referencia` é constante e, portanto, não
> carrega informação discriminante.

---

## 3. Variáveis categóricas de segmentação

| Variável | Cardinalidade | Categorias |
|---|---|---|
| `setor_economico` | 12 | Agronegócio, Bancos e Serviços Financeiros, Bens de Consumo, Construção Civil, Energia, Indústria, Logística e Transportes, Saúde, Serviços Empresariais, Tecnologia, Telecomunicações, Varejo |
| `subsetor` | 48 | 4 subsetores por setor (ver tabela abaixo) |
| `pais` | 10 | Estados Unidos, Brasil, Índia, Alemanha, Reino Unido, México, Canadá, Japão, França, Austrália |
| `regiao` | 5 | América do Norte, América Latina, Europa, Ásia, Oceania |
| `porte_empresa` | 3 | Pequena, Média, Grande |
| `estagio_maturidade` | 4 | Inicial, Expansão, Madura, Reestruturação |

### Distribuição

**Setor econômico** — distribuição balanceada por construção: 666 ou 667 empresas por setor.

**Região:** América do Norte 2.535 · América Latina 2.031 · Europa 1.732 · Ásia 1.270 · Oceania 432

**Porte:** Média 3.171 · Pequena 2.775 · Grande 2.054

**Estágio de maturidade:** Madura 3.920 · Expansão 2.012 · Reestruturação 1.269 · Inicial 799

### Hierarquia setor → subsetor

| Setor | Subsetores |
|---|---|
| Agronegócio | Alimentos e Insumos, Produção Agrícola, Proteína Animal, Trading Agrícola |
| Bancos e Serviços Financeiros | Banco Comercial, Fintech, Gestão de Ativos, Seguros |
| Bens de Consumo | Alimentos e Bebidas, Bens Duráveis, Higiene e Beleza, Produtos Domésticos |
| Construção Civil | Engenharia, Incorporação, Infraestrutura, Materiais de Construção |
| Energia | Energia Elétrica, Petróleo e Gás, Renováveis, Serviços de Energia |
| Indústria | Automotivo, Materiais Industriais, Máquinas e Equipamentos, Química |
| Logística e Transportes | Aviação, Operador Logístico, Portos e Terminais, Transporte Rodoviário |
| Saúde | Diagnósticos, Equipamentos Médicos, Farmacêutica, Hospitais |
| Serviços Empresariais | Consultoria, Educação Corporativa, Serviços Profissionais, Terceirização |
| Tecnologia | Plataformas Digitais, Semicondutores, Serviços de TI, Software SaaS |
| Telecomunicações | Banda Larga, Infraestrutura de Telecom, Serviços Digitais, Telefonia Móvel |
| Varejo | E-commerce, Moda e Vestuário, Supermercados, Varejo Especializado |

---

## 4. Variáveis financeiras

Todos os valores monetários estão em **milhões de dólares (USD mi)**, indicados pelo
sufixo `_usd_m`. O sufixo `_pct` indica percentual e `_t_1` / `_t_2` indicam defasagem
de um e dois períodos.

### 4.1 Receita e crescimento

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `receita_bruta_usd_m` | float | 0,91 | 132,61 | 92.871,46 | Receita antes de deduções |
| `receita_liquida_usd_m` | float | 0,86 | 126,65 | 87.847,46 | Receita após deduções e impostos sobre vendas |
| `receita_liquida_t_1_usd_m` | float | 0,64 | 119,16 | 86.590,36 | Receita líquida do período anterior |
| `receita_liquida_t_2_usd_m` | float | 0,52 | 111,20 | 95.715,96 | Receita líquida de dois períodos atrás |
| `receita_recorrente_usd_m` | float | 0,00 | 54,08 | 76.482,04 | Parcela contratual/recorrente da receita |
| `receita_internacional_usd_m` | float | 0,00 | 25,53 | 39.864,39 | Receita originada fora do país-sede |
| `crescimento_receita_pct` | float | -40,97 | 7,43 | 54,27 | Variação percentual da receita sobre t-1 |
| `taxa_crescimento_anual_pct` | float | -36,18 | 6,99 | 45,73 | Taxa de crescimento anualizada |

### 4.2 Resultado e rentabilidade

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `lucro_bruto_usd_m` | float | -1.889,98 | 43,29 | 55.421,64 | Receita líquida menos custo dos produtos vendidos |
| `ebitda_usd_m` | float | -3.552,71 | 17,21 | 25.348,57 | Resultado antes de juros, impostos, depreciação e amortização |
| `margem_ebitda_pct` | float | -19,83 | 15,97 | 50,93 | EBITDA sobre receita líquida |
| `lucro_operacional_usd_m` | float | -5.928,97 | 10,53 | 19.482,48 | Resultado da atividade-fim (EBIT) |
| `margem_operacional_pct` | float | -22,92 | 11,83 | 43,81 | Lucro operacional sobre receita líquida |
| `lucro_liquido_usd_m` | float | -39.266,51 | 2,95 | 14.664,78 | Resultado final do exercício |
| `margem_liquida_pct` | float | -116,17 | 5,53 | 30,14 | Lucro líquido sobre receita líquida |
| `margem_bruta_pct` | float | -7,63 | 36,69 | 90,00 | Lucro bruto sobre receita líquida |
| `roa_pct` | float | -47,10 | 4,51 | 31,59 | Retorno sobre ativos totais |
| `roe_pct` | float | -384,26 | 13,81 | 103,56 | Retorno sobre patrimônio líquido |
| `roi_pct` | float | -44,07 | 9,93 | 44,16 | Retorno sobre o capital investido |

### 4.3 Estrutura patrimonial e endividamento

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `ativos_totais_usd_m` | float | 0,47 | 164,77 | 958.857,97 | Total do ativo |
| `passivos_totais_usd_m` | float | 0,37 | 103,51 | 938.426,40 | Total do passivo (obrigações) |
| `divida_curto_prazo_usd_m` | float | 0,04 | 17,35 | 93.065,34 | Dívida onerosa com vencimento em até 12 meses |
| `divida_longo_prazo_usd_m` | float | 0,08 | 31,63 | 147.302,22 | Dívida onerosa com vencimento acima de 12 meses |
| `divida_total_usd_m` | float | 0,16 | 50,56 | 219.690,90 | Soma da dívida de curto e longo prazo |
| `patrimonio_liquido_usd_m` | float | 0,10 | 46,60 | 81.149,29 | Capital próprio (ativos menos passivos) |
| `debt_to_equity` | float | 0,13 | 1,02 | 18,00 | Dívida total sobre patrimônio líquido |
| `indice_alavancagem` | float | 1,51 | 2,95 | 83,33 | Ativos totais sobre patrimônio líquido |
| `cobertura_juros` | float | -5,38 | 2,95 | 87,47 | Capacidade de o resultado operacional cobrir despesas financeiras |

### 4.4 Liquidez e capital de giro

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `ativos_circulantes_usd_m` | float | 0,03 | 44,04 | 512.225,90 | Ativos conversíveis em caixa em até 12 meses |
| `passivos_circulantes_usd_m` | float | 0,05 | 33,64 | 396.172,57 | Obrigações vencíveis em até 12 meses |
| `caixa_usd_m` | float | 0,00 | 8,53 | 69.869,49 | Disponibilidades imediatas |
| `equivalentes_caixa_usd_m` | float | 0,00 | 3,86 | 31.339,73 | Aplicações de altíssima liquidez |
| `capital_giro_usd_m` | float | -90.329,92 | 3,86 | 136.944,34 | Ativos circulantes menos passivos circulantes |
| `liquidez_corrente` | float | 0,36 | 1,30 | 2,97 | Ativos circulantes sobre passivos circulantes |
| `liquidez_seca` | float | 0,14 | 0,76 | 2,84 | Liquidez corrente excluindo estoques |
| `liquidez_imediata` | float | 0,02 | 0,41 | 1,86 | Caixa e equivalentes sobre passivos circulantes |

### 4.5 Fluxo de caixa

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `fluxo_operacional_usd_m` | float | -46.289,82 | 7,37 | 19.555,18 | Caixa gerado pela operação |
| `fluxo_investimentos_usd_m` | float | -13.999,65 | -6,86 | -0,01 | Caixa aplicado em investimentos (sempre negativo) |
| `fluxo_financiamentos_usd_m` | float | -19.663,74 | 0,68 | 42.749,55 | Caixa de captações e amortizações |
| `fluxo_caixa_livre_usd_m` | float | -46.918,46 | 0,61 | 13.802,40 | Fluxo operacional menos capex |
| `fluxo_caixa_livre_t_1_usd_m` | float | -50.017,03 | 0,58 | 17.960,69 | Fluxo de caixa livre do período anterior |
| `variacao_caixa_usd_m` | float | -20.897,76 | 2,83 | 44.782,58 | Variação líquida do saldo de caixa no período |

### 4.6 Custos, despesas e investimentos

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `custos_operacionais_usd_m` | float | 0,64 | 92,92 | 68.223,08 | Custos diretamente ligados à produção/serviço |
| `despesas_administrativas_usd_m` | float | 0,04 | 9,86 | 12.102,66 | Despesas gerais e administrativas |
| `despesas_financeiras_usd_m` | float | 0,01 | 4,56 | 43.731,12 | Juros e encargos sobre dívida |
| `investimentos_capex_usd_m` | float | 0,01 | 7,34 | 14.402,10 | Investimento em ativo imobilizado |
| `opex_usd_m` | float | 0,71 | 104,22 | 80.168,85 | Despesa operacional total |

### 4.7 Eficiência operacional

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `giro_ativo` | float | 0,04 | 0,88 | 3,59 | Receita líquida sobre ativos totais |
| `giro_estoque` | float | 0,00 | 4,56 | 14,68 | Número de renovações do estoque no período |
| `prazo_medio_recebimento_dias` | float | 0,00 | 47,88 | 120,25 | Dias médios para receber de clientes |
| `prazo_medio_pagamento_dias` | float | 0,00 | 49,00 | 126,64 | Dias médios para pagar fornecedores |

### 4.8 Porte e posicionamento de mercado

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição |
|---|---|---|---|---|---|
| `valor_mercado_estimado_usd_m` | float | 0,23 | 178,14 | 325.701,53 | Valuation estimado da empresa |
| `numero_funcionarios` | inteira | 10 | 249 | 557.830 | Quadro de colaboradores |
| `market_share_estimado_pct` | float | 0,00 | 1,57 | 28,00 | Participação estimada no mercado do subsetor |
| `participacao_internacional_pct` | float | 0,00 | 27,18 | 89,43 | Percentual da receita originada no exterior |

### 4.9 Índices sintéticos

Índices compostos, gerados pelo próprio sistema idealizado a partir das demais variáveis.

| Variável | Tipo | Mín. | Mediana | Máx. | Descrição | Usado como atributo? |
|---|---|---|---|---|---|---|
| `indice_solvencia` | float | 1,01 | 1,51 | 2,95 | Capacidade estrutural de honrar obrigações | Sim (baseline) |
| `indice_cobertura` | float | -2,97 | 2,56 | 33,89 | Cobertura consolidada de compromissos financeiros | Sim (baseline) |
| `indice_eficiencia_0_100` | float | 32,18 | 53,10 | 65,33 | Eficiência operacional normalizada em escala 0–100 | Sim (regressão logística) |
| `indice_risco_financeiro_0_100` | float | 1,00 | 20,44 | 99,00 | Risco financeiro consolidado (0 = mínimo) | **Não — excluído** |
| `indice_saude_financeira_0_100` | float | 1,00 | 79,68 | 99,00 | Saúde financeira consolidada (100 = máxima) | **Não — excluído** |
| `score_financeiro_geral_0_100` | float | 0,00 | 68,05 | 93,01 | Score agregado de qualidade financeira | **Não — excluído** |

> **Decisão metodológica — prevenção de vazamento.** As três últimas variáveis foram
> construídas pelo gerador do dataset em conjunto com as variáveis-alvo e apresentam
> correlação de aproximadamente 0,45 em módulo com `target_deterioracao_financeira`.
> Utilizá-las como atributos equivaleria a fornecer ao modelo uma versão indireta da
> resposta. Elas estão declaradas em `SYNTHETIC_RISK_COLUMNS`
> (`src/features/feature_catalog.py`) e removidas de todos os conjuntos de atributos.

---

## 5. Variáveis-alvo

O dataset disponibiliza oito alvos. O projeto utiliza dois de forma supervisionada;
os demais permanecem disponíveis para extensões e **nunca entram como atributos**
(ver `TARGET_COLUMNS` em `src/domain/dataset_schema.py`).

| Variável | Tipo | Domínio / faixa | Distribuição | Uso no projeto |
|---|---|---|---|---|
| `target_deterioracao_financeira` | Categórica binária | `Sim`, `Não` | Não 6.201 (77,5%) · Sim 1.799 (22,5%) | **Classificador — regressão logística** |
| `target_fluxo_caixa_livre_proximo_periodo_usd_m` | Contínua | -48.904,11 a 11.056,47 | Mediana 0,74 | **Alvo — regressão linear** |
| `target_probabilidade_deterioracao_pct` | Contínua | 0,50 a 99,50 | Mediana 2,65 | Não utilizado |
| `target_classe_risco` | Categórica ordinal | `Baixo`, `Médio`, `Alto` | Baixo 5.514 · Alto 1.891 · Médio 595 | Não utilizado |
| `target_faixa_saude_financeira` | Categórica ordinal | `Crítica`, `Atenção`, `Boa`, `Excelente` | Excelente 4.103 · Crítica 2.425 · Boa 785 · Atenção 687 | Não utilizado |
| `target_probabilidade_insolvencia_pct` | Contínua | 0,10 a 98,00 | Mediana 1,12 | Não utilizado |
| `target_tende_crescer_proximo_periodo` | Categórica binária | `Sim`, `Não` | Não 4.102 · Sim 3.898 | Não utilizado |
| `target_crescimento_receita_proximo_periodo_pct` | Contínua | -38,25 a 47,47 | Mediana 7,05 | Não utilizado |

### Classificador principal

`target_deterioracao_financeira` indica se a empresa apresentará deterioração de sua
condição financeira no período seguinte. A classe positiva (`Sim`) representa **22,5%**
das observações, caracterizando **desbalanceamento moderado**. Essa proporção justifica
duas decisões do projeto:

1. A seleção de hiperparâmetros usa **PR-AUC** (`average_precision`) como métrica de
   otimização, e não acurácia, que seria de 77,5% para um classificador trivial.
2. O grid de busca inclui `class_weight ∈ {None, 'balanced'}`, e a calibração sigmoide
   posterior corrige o deslocamento de probabilidades introduzido pelo reponderamento.

O K-Means, por ser **não supervisionado**, não consome nenhuma variável-alvo: os cinco
clusters emergem exclusivamente da estrutura dos atributos financeiros.

---

## 6. Variáveis derivadas

Razões financeiras calculadas em tempo de execução — **não existem no CSV**. São
produzidas por `add_financial_ratios` (`src/features/financial_ratios.py`) e pelos
transformadores dos pipelines, sempre com divisão protegida contra denominador nulo
(`safe_divide`), que converte resultados infinitos em ausentes.

| Variável derivada | Fórmula | Origem |
|---|---|---|
| `margem_bruta_calculada` | lucro bruto ÷ receita líquida | `add_financial_ratios` |
| `margem_ebitda_calculada` | EBITDA ÷ receita líquida | `add_financial_ratios` |
| `margem_liquida_calculada` | lucro líquido ÷ receita líquida | `add_financial_ratios` |
| `divida_sobre_ativos` | dívida total ÷ ativos totais | `add_financial_ratios` |
| `caixa_sobre_passivos_circulantes` | caixa ÷ passivos circulantes | `add_financial_ratios` |
| `fluxo_operacional_sobre_receita` | fluxo operacional ÷ receita líquida | `add_financial_ratios` |
| `margem_fluxo_caixa_livre_calculada` | fluxo de caixa livre ÷ receita líquida | `add_financial_ratios` |
| `margem_fcf_atual` | fluxo de caixa livre ÷ receita líquida | `CashflowFeatureTransformer` |
| `capital_giro_sobre_receita` | capital de giro ÷ receita líquida | `CashflowFeatureTransformer` |
| `divida_sobre_receita` | dívida total ÷ receita líquida | `CashflowFeatureTransformer` |
| `cluster_kmeans` | rótulo do K-Means treinado (0–4) | Artefato `kmeans_compact_v2.0.0.joblib` |

> **Racional.** Razões substituem valores monetários absolutos porque estes fazem os
> algoritmos agruparem e prever por **porte** da empresa, e não por **comportamento
> financeiro** — efeito observado nos testes iniciais do experimento `baseline`.

---

## 7. Conjuntos de atributos por algoritmo

### K-Means — experimento `compact` (modelo adotado, k=5)

Oito atributos numéricos, sem variáveis categóricas e sem valores monetários absolutos:

`crescimento_receita_pct` · `margem_ebitda_pct` · `margem_liquida_pct` ·
`divida_sobre_ativos` · `cobertura_juros` · `liquidez_corrente` ·
`margem_fluxo_caixa_livre_calculada` · `giro_ativo`

### K-Means — experimento `baseline` (comparação)

Todas as variáveis financeiras exceto os três índices sintéticos excluídos, acrescidas
das razões derivadas e das seis variáveis categóricas com codificação one-hot.

### Regressão logística — deterioração financeira

Doze atributos numéricos mais o cluster do K-Means como variável contextual:

`crescimento_receita_pct` · `taxa_crescimento_anual_pct` · `margem_ebitda_pct` ·
`margem_liquida_pct` · `roa_pct` · `debt_to_equity` · `cobertura_juros` ·
`liquidez_corrente` · `liquidez_imediata` · `fluxo_caixa_livre_usd_m` ·
`capital_giro_usd_m` · `indice_eficiencia_0_100` · `cluster_kmeans`

### Regressão linear — fluxo de caixa livre futuro

Cinco razões financeiras mais o cluster:

`crescimento_receita_pct` · `margem_ebitda_pct` · `margem_fcf_atual` ·
`capital_giro_sobre_receita` · `divida_sobre_receita` · `cluster_kmeans`

---

## 8. Tratamento aplicado aos dados

| Etapa | Método | Observação |
|---|---|---|
| Valores ausentes | Mediana (numéricas) / moda (categóricas) | O CSV não possui ausentes; a imputação protege razões derivadas que geram `NaN` por denominador nulo |
| Valores extremos | Winsorização nos percentis 1 e 99 (`QuantileClipper`) | Limites aprendidos **apenas no treino** e reaplicados em validação e teste |
| Escala | `StandardScaler` (média 0, desvio 1) | Necessário para K-Means (distância euclidiana) e para interpretar coeficientes da logística |
| Categóricas | `OneHotEncoder(handle_unknown='ignore')` | `drop='first'` na logística, definindo um cluster de referência |

Todas as transformações são etapas de um `Pipeline` do scikit-learn, garantindo que
nenhuma estatística das partições de validação ou teste influencie o ajuste.

---

## 9. Referência rápida — todas as 79 colunas

| # | Coluna | Tipo | Grupo |
|---|---|---|---|
| 1 | `id_empresa` | Categórica | Identificador |
| 2 | `nome_empresa_ficticio` | Categórica | Identificador |
| 3 | `setor_economico` | Categórica | Segmentação |
| 4 | `subsetor` | Categórica | Segmentação |
| 5 | `pais` | Categórica | Segmentação |
| 6 | `regiao` | Categórica | Segmentação |
| 7 | `ano_referencia` | Inteira | Metadado |
| 8 | `porte_empresa` | Categórica | Segmentação |
| 9 | `estagio_maturidade` | Categórica | Segmentação |
| 10 | `split_ml` | Categórica | Metadado |
| 11 | `receita_bruta_usd_m` | Contínua | Receita |
| 12 | `receita_liquida_usd_m` | Contínua | Receita |
| 13 | `receita_liquida_t_1_usd_m` | Contínua | Receita |
| 14 | `receita_liquida_t_2_usd_m` | Contínua | Receita |
| 15 | `receita_recorrente_usd_m` | Contínua | Receita |
| 16 | `receita_internacional_usd_m` | Contínua | Receita |
| 17 | `crescimento_receita_pct` | Contínua | Receita |
| 18 | `taxa_crescimento_anual_pct` | Contínua | Receita |
| 19 | `lucro_bruto_usd_m` | Contínua | Rentabilidade |
| 20 | `ebitda_usd_m` | Contínua | Rentabilidade |
| 21 | `margem_ebitda_pct` | Contínua | Rentabilidade |
| 22 | `lucro_operacional_usd_m` | Contínua | Rentabilidade |
| 23 | `margem_operacional_pct` | Contínua | Rentabilidade |
| 24 | `lucro_liquido_usd_m` | Contínua | Rentabilidade |
| 25 | `margem_liquida_pct` | Contínua | Rentabilidade |
| 26 | `margem_bruta_pct` | Contínua | Rentabilidade |
| 27 | `roa_pct` | Contínua | Rentabilidade |
| 28 | `roe_pct` | Contínua | Rentabilidade |
| 29 | `roi_pct` | Contínua | Rentabilidade |
| 30 | `ativos_totais_usd_m` | Contínua | Estrutura patrimonial |
| 31 | `passivos_totais_usd_m` | Contínua | Estrutura patrimonial |
| 32 | `divida_curto_prazo_usd_m` | Contínua | Endividamento |
| 33 | `divida_longo_prazo_usd_m` | Contínua | Endividamento |
| 34 | `divida_total_usd_m` | Contínua | Endividamento |
| 35 | `patrimonio_liquido_usd_m` | Contínua | Estrutura patrimonial |
| 36 | `debt_to_equity` | Contínua | Endividamento |
| 37 | `indice_alavancagem` | Contínua | Endividamento |
| 38 | `cobertura_juros` | Contínua | Endividamento |
| 39 | `ativos_circulantes_usd_m` | Contínua | Liquidez |
| 40 | `passivos_circulantes_usd_m` | Contínua | Liquidez |
| 41 | `caixa_usd_m` | Contínua | Liquidez |
| 42 | `equivalentes_caixa_usd_m` | Contínua | Liquidez |
| 43 | `capital_giro_usd_m` | Contínua | Liquidez |
| 44 | `liquidez_corrente` | Contínua | Liquidez |
| 45 | `liquidez_seca` | Contínua | Liquidez |
| 46 | `liquidez_imediata` | Contínua | Liquidez |
| 47 | `fluxo_operacional_usd_m` | Contínua | Fluxo de caixa |
| 48 | `fluxo_investimentos_usd_m` | Contínua | Fluxo de caixa |
| 49 | `fluxo_financiamentos_usd_m` | Contínua | Fluxo de caixa |
| 50 | `fluxo_caixa_livre_usd_m` | Contínua | Fluxo de caixa |
| 51 | `fluxo_caixa_livre_t_1_usd_m` | Contínua | Fluxo de caixa |
| 52 | `variacao_caixa_usd_m` | Contínua | Fluxo de caixa |
| 53 | `custos_operacionais_usd_m` | Contínua | Custos e despesas |
| 54 | `despesas_administrativas_usd_m` | Contínua | Custos e despesas |
| 55 | `despesas_financeiras_usd_m` | Contínua | Custos e despesas |
| 56 | `investimentos_capex_usd_m` | Contínua | Custos e despesas |
| 57 | `opex_usd_m` | Contínua | Custos e despesas |
| 58 | `giro_ativo` | Contínua | Eficiência |
| 59 | `giro_estoque` | Contínua | Eficiência |
| 60 | `prazo_medio_recebimento_dias` | Contínua | Eficiência |
| 61 | `prazo_medio_pagamento_dias` | Contínua | Eficiência |
| 62 | `valor_mercado_estimado_usd_m` | Contínua | Mercado |
| 63 | `numero_funcionarios` | Inteira | Mercado |
| 64 | `market_share_estimado_pct` | Contínua | Mercado |
| 65 | `participacao_internacional_pct` | Contínua | Mercado |
| 66 | `indice_solvencia` | Contínua | Índice sintético |
| 67 | `indice_cobertura` | Contínua | Índice sintético |
| 68 | `indice_eficiencia_0_100` | Contínua | Índice sintético |
| 69 | `indice_risco_financeiro_0_100` | Contínua | Índice sintético (excluído) |
| 70 | `indice_saude_financeira_0_100` | Contínua | Índice sintético (excluído) |
| 71 | `score_financeiro_geral_0_100` | Contínua | Índice sintético (excluído) |
| 72 | `target_deterioracao_financeira` | Categórica | Alvo |
| 73 | `target_probabilidade_deterioracao_pct` | Contínua | Alvo |
| 74 | `target_classe_risco` | Categórica | Alvo |
| 75 | `target_faixa_saude_financeira` | Categórica | Alvo |
| 76 | `target_probabilidade_insolvencia_pct` | Contínua | Alvo |
| 77 | `target_tende_crescer_proximo_periodo` | Categórica | Alvo |
| 78 | `target_fluxo_caixa_livre_proximo_periodo_usd_m` | Contínua | Alvo |
| 79 | `target_crescimento_receita_proximo_periodo_pct` | Contínua | Alvo |

---

## 10. Limitações do dataset

- **Dados sintéticos.** As relações entre variáveis foram geradas artificialmente. As
  métricas obtidas não devem ser extrapoladas para carteiras reais de crédito.
- **Corte transversal único.** Todas as observações referem-se a 2025. As defasagens
  disponíveis (`t_1`, `t_2`) são pontuais e não permitem modelagem de séries temporais.
- **Ausência total de valores nulos.** Condição irrealista frente a bases financeiras
  reais; a imputação foi mantida no pipeline por robustez operacional.
- **Sobrerrepresentação de América do Norte e América Latina** (57% das instâncias),
  o que limita a generalização das conclusões por região.
