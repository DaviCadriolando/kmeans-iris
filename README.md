# Atividade 9 – Aprendizagem Não Supervisionada (K-Means na base Iris)

Atividade da disciplina de Sistemas de Informação / IA (UCB – Análise e Desenvolvimento de Sistemas).
O objetivo é agrupar flores da base **Iris** usando **K-Means**, **sem** usar o rótulo da espécie, e depois comparar os grupos encontrados com as espécies reais.

**Autor:** Davi Rodrigues dos Santos Cariolano ([@TrevoCode](https://github.com/TrevoCode))

---

## O que o projeto faz

1. Carrega e explora a base Iris (150 flores, 4 medidas numéricas).
2. Limpa e valida os dados (ausentes, duplicados, valores ≤ 0).
3. Mostra como funciona a codificação (One-Hot) com um exemplo fictício.
4. Padroniza as características com `StandardScaler`.
5. Faz análise exploratória (histogramas, correlação, boxplots, dispersões).
6. Aplica o K-Means: escolhe K com **Método do Cotovelo** e **Silhouette Score**, treina com K = 3 e visualiza com **PCA**.
7. Compara K = 2, 3 e 4 e cruza os grupos com as espécies reais.

## Estrutura do repositório

```
atividade-9-kmeans-iris/
├── notebook.ipynb      # Notebook completo (código + análises + conclusão)
├── data/
│   └── README.md       # Onde baixar o Iris.csv (arquivo não versionado)
├── requirements.txt    # Dependências
└── README.md
```

## Como executar

### Opção 1 – Google Colab (recomendada)
1. Acesse [colab.research.google.com](https://colab.research.google.com) → *Arquivo → Fazer upload de notebook* → `notebook.ipynb`.
2. Baixe o `Iris.csv` no [Kaggle](https://www.kaggle.com/datasets/uciml/iris).
3. Rode o notebook; a primeira célula de dados abre o upload do CSV se ele não estiver no ambiente.
4. *Ambiente de execução → Executar tudo*.

### Opção 2 – Local
```bash
git clone https://github.com/TrevoCode/atividade-9-kmeans-iris.git
cd atividade-9-kmeans-iris
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
# coloque o Iris.csv na pasta do notebook e rode:
jupyter notebook notebook.ipynb
```
> No notebook, o caminho lido é `Iris.csv` (na mesma pasta do notebook).

## Decisões principais

| Decisão | Motivo |
|---|---|
| Remover `Id` e `Species` do agrupamento | `Id` é só identificador; `Species` é o rótulo real. K-Means é não supervisionado e usar o rótulo seria vazamento de informação. |
| Não remover duplicados | Medidas iguais não são necessariamente erro; cada linha é uma flor diferente. |
| Padronizar com `StandardScaler` | K-Means usa distância euclidiana, então todas as variáveis precisam estar na mesma escala. |
| `random_state=42` e `n_init=10` | Resultados reproduzíveis e mais estáveis. |
| PCA só para visualização | Os grupos vêm do K-Means treinado com as 4 variáveis padronizadas; o PCA apenas projeta em 2D. |

## Resultados esperados

> Valores obtidos em uma execução de teste com `random_state=42`. Podem variar levemente dependendo da versão do CSV e das bibliotecas; confira no notebook após rodar.

| K | Inércia | Silhouette |
|---|---------|------------|
| 2 | ≈ 222 | ≈ 0,58 |
| 3 | ≈ 140 | ≈ 0,46 |
| 4 | ≈ 114 | ≈ 0,39 |

- Maior correlação: comprimento × largura da **pétala** (≈ 0,96).
- PCA: duas componentes explicam ≈ 96% da variância.
- Com K = 3, a *Iris setosa* fica 100% isolada; *versicolor* e *virginica* se misturam um pouco na fronteira, pois suas medidas se sobrepõem.
- **Conclusão:** K = 2 tem a melhor métrica de silhouette, mas K = 3 é a configuração mais interpretável e a que mais se aproxima das espécies reais.

## Tecnologias

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Google Colab
