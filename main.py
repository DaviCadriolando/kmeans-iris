# ==================== ETAPA 1 - PREPARAÇÃO ====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
SEED = 42

if not os.path.exists("Iris.csv"):
    from google.colab import files
    files.upload()  # selecione o Iris.csv

df = pd.read_csv("Iris.csv")
display(df.head())

# ==================== ETAPA 2 - SELEÇÃO ====================
print("Registros:", df.shape[0])
print("Colunas:", df.shape[1])
print(df.dtypes)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[features].copy()   # Id e Species ficam de fora
print("Formato de X:", X.shape)
display(X.head())
X.info()

# ==================== ETAPA 3 - LIMPEZA ====================
print("Ausentes por coluna:\n", df.isnull().sum())

dup = X.duplicated(keep=False)
print("Linhas com features repetidas:", dup.sum())
print("Duplicatas excedentes:", X.duplicated().sum())
display(df.loc[dup, ["Id"] + features + ["Species"]])

print(X.dtypes)
print("Medidas <= 0:\n", (X <= 0).sum())
display(X.describe())
# Sem tratamento necessário. Repetidos mantidos (medidas iguais != erro).

# ==================== ETAPA 4 - CODIFICAÇÃO ====================
print("Colunas categóricas:", df.select_dtypes(exclude="number").columns.tolist())
print("Features são numéricas?", X.select_dtypes(include=np.number).shape[1] == 4)

fake = pd.DataFrame({
    "cor_da_flor": ["vermelha", "azul", "amarela", "azul", "vermelha"],
    "altura_cm": [30, 25, 40, 28, 35]
})
print("Antes:"); display(fake)
print("Depois do One-Hot Encoding:")
display(pd.get_dummies(fake, columns=["cor_da_flor"], dtype=int))

# ==================== ETAPA 5 - PADRONIZAÇÃO ====================
antes = pd.DataFrame({"média": X.mean(), "desvio padrão": X.std()})
print("ANTES:"); display(antes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
display(X_scaled_df.head())

depois = pd.DataFrame({"média": X_scaled_df.mean(),
                       "desvio padrão (ddof=0)": X_scaled_df.std(ddof=0)})
print("DEPOIS:"); display(depois.round(6))

# ==================== ETAPA 6 - EXPLORAÇÃO ====================
# 6.1 Histogramas
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(df[col], bins=15, kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribuição de {col}")
    ax.set_xlabel(col + " (cm)"); ax.set_ylabel("Frequência")
plt.tight_layout(); plt.show()

# 6.2 Matriz de correlação
plt.figure(figsize=(7, 5.5))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Matriz de correlação"); plt.show()

# 6.3 Boxplots
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, col in zip(axes, features):
    sns.boxplot(y=df[col], ax=ax, color="lightseagreen")
    ax.set_title(col); ax.set_ylabel("cm")
plt.suptitle("Boxplots das características", y=1.02)
plt.tight_layout(); plt.show()

for c in features:
    q1, q3 = X[c].quantile([0.25, 0.75]); iqr = q3 - q1
    n = ((X[c] < q1 - 1.5*iqr) | (X[c] > q3 + 1.5*iqr)).sum()
    print(f"{c}: {n} outlier(s)")

# 6.4 Dispersão sépalas e pétalas
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(df["SepalLengthCm"], df["SepalWidthCm"], alpha=0.7)
axes[0].set_title("Sépalas: comprimento x largura")
axes[0].set_xlabel("Comprimento da sépala (cm)"); axes[0].set_ylabel("Largura da sépala (cm)")
axes[1].scatter(df["PetalLengthCm"], df["PetalWidthCm"], alpha=0.7, color="darkorange")
axes[1].set_title("Pétalas: comprimento x largura")
axes[1].set_xlabel("Comprimento da pétala (cm)"); axes[1].set_ylabel("Largura da pétala (cm)")
plt.tight_layout(); plt.show()

# 6.5 Pétalas padronizadas
plt.figure(figsize=(6.5, 5))
plt.scatter(X_scaled_df["PetalLengthCm"], X_scaled_df["PetalWidthCm"], alpha=0.7, color="purple")
plt.title("Pétalas padronizadas: comprimento x largura")
plt.xlabel("Comprimento da pétala (padronizado)"); plt.ylabel("Largura da pétala (padronizada)")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.show()

# ==================== ETAPA 7 - K-MEANS ====================
# ---- Parte A: Cotovelo e Silhouette
ks = range(1, 11)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled).inertia_ for k in ks]

plt.figure(figsize=(7, 4.5))
plt.plot(list(ks), inertias, marker="o")
plt.title("Método do Cotovelo"); plt.xlabel("Número de grupos (K)"); plt.ylabel("Inércia")
plt.xticks(list(ks)); plt.show()
display(pd.DataFrame({"K": list(ks), "Inércia": inertias}).round(2))

ks_sil = range(2, 11)
sil_scores = []
for k in ks_sil:
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, lab))

plt.figure(figsize=(7, 4.5))
plt.plot(list(ks_sil), sil_scores, marker="o", color="darkorange")
plt.title("Silhouette Score por K"); plt.xlabel("Número de grupos (K)"); plt.ylabel("Silhouette Score")
plt.xticks(list(ks_sil)); plt.show()
display(pd.DataFrame({"K": list(ks_sil), "Silhouette": sil_scores}).round(4))

# ---- Parte B: K = 3
kmeans3 = KMeans(n_clusters=3, n_init=10, random_state=SEED)
df["Cluster"] = kmeans3.fit_predict(X_scaled)
print("Observações por grupo:\n", df["Cluster"].value_counts().sort_index())

plt.figure(figsize=(7, 5.5))
sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Cluster", palette="Set1", s=60)
plt.title("K-Means (K=3): comprimento x largura da pétala")
plt.xlabel("Comprimento da pétala (cm)"); plt.ylabel("Largura da pétala (cm)")
plt.legend(title="Grupo"); plt.show()

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
print("Variância explicada:", pca.explained_variance_ratio_.round(4),
      "| total:", pca.explained_variance_ratio_.sum().round(4))

plt.figure(figsize=(7, 5.5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["Cluster"], palette="Set1", s=60)
plt.title("Grupos do K-Means (K=3) em 2 componentes (PCA)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.legend(title="Grupo"); plt.show()

perfil = df.groupby("Cluster")[features].mean()
display(perfil.round(3))

plt.figure(figsize=(8, 4))
sns.heatmap(perfil, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Perfil médio dos grupos (K=3)"); plt.ylabel("Grupo"); plt.show()

# ---- Parte C: K = 2, 3 e 4
resultados, labels_por_k = [], {}
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    lab = km.fit_predict(X_scaled)
    labels_por_k[k] = lab
    resultados.append({"K": k, "Inércia": km.inertia_,
                       "Silhouette": silhouette_score(X_scaled, lab)})
display(pd.DataFrame(resultados).round(4))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, k in zip(axes, [2, 3, 4]):
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_por_k[k],
                    palette="Set1", s=50, ax=ax)
    ax.set_title(f"K = {k} (PCA)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(title="Grupo")
plt.tight_layout(); plt.show()

# Comparação com as espécies reais (K=3)
tab = pd.crosstab(df["Species"], df["Cluster"],
                  rownames=["Espécie real"], colnames=["Grupo K-Means"])
display(tab)
tab.plot(kind="bar", figsize=(8, 5), colormap="Set1")
plt.title("Espécies reais x grupos do K-Means (K=3)")
plt.xlabel("Espécie"); plt.ylabel("Quantidade de flores")
plt.xticks(rotation=0); plt.legend(title="Grupo"); plt.show()

# Extra: K=2
display(pd.crosstab(df["Species"], labels_por_k[2],
                    rownames=["Espécie real"], colnames=["Grupo (K=2)"]))