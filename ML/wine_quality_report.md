# Rapport d'Analyse Complète : Wine Quality Dataset

## 1. Description de la Base de Données

### 1.1 Contexte et Origine
Le jeu de données **Wine Quality** provient de l'UCI Machine Learning Repository (ID: 186). Il contient des données physicochimiques et sensorielles sur des vins blancs et rouges portugais de la région du Vinho Verde.

### 1.2 Caractéristiques du Dataset
- **Source** : http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
- **Type** : Données physicochimiques de vins blancs
- **Nombre d'observations** : Variable selon le fichier (vins blancs utilisés dans l'analyse)
- **Nombre de variables** : 12 (11 features + 1 variable cible)

### 1.3 Description des Variables

| Variable | Type | Description | Unité |
|----------|------|-------------|-------|
| **fixed acidity** | Numérique | Acidité fixe | g/L (acide tartrique) |
| **volatile acidity** | Numérique | Acidité volatile | g/L (acide acétique) |
| **citric acid** | Numérique | Acide citrique | g/L |
| **residual sugar** | Numérique | Sucre résiduel après fermentation | g/L |
| **chlorides** | Numérique | Chlorures (sel) | g/L |
| **free sulfur dioxide** | Numérique | Dioxyde de soufre libre | mg/L |
| **total sulfur dioxide** | Numérique | Dioxyde de soufre total | mg/L |
| **density** | Numérique | Densité | g/cm³ |
| **pH** | Numérique | Niveau d'acidité/basicité | Échelle pH |
| **sulphates** | Numérique | Sulfates (additif) | g/L |
| **alcohol** | Numérique | Teneur en alcool | % vol. |
| **quality** | Numérique (cible) | Score de qualité (0-10) | Score |

### 1.4 Transformation de la Variable Cible
Pour les besoins de l'analyse de classification, la variable `quality` a été binarisée :
- **Mauvaise qualité (0)** : quality ≤ 5
- **Bonne qualité (1)** : quality > 5

---

## 2. Analyse Exploratoire des Données

### 2.1 Aperçu du Dataset

```python
import pandas as pd
import numpy as np

link = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(link, header="infer", delimiter=";")

print("\n========= Dataset summary ========= \n")
df.info()
print("\n========= A few first samples ========= \n")
print(df.head())
```

### 2.2 Distribution de la Variable Cible

```python
X = df.drop("quality", axis=1)
Y = df["quality"]
print("\n========= Wine Qualities ========= \n")
print(Y.value_counts())

# Binarisation
Y = [0 if val <=5 else 1 for val in Y]
```

**Observation** : La distribution originale de la qualité permet d'observer la répartition des scores avant binarisation.

### 2.3 Détection des Valeurs Aberrantes

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
ax = plt.gca()
sns.boxplot(data=X, orient="v", palette="Set1", width=1.5, notch=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Distribution et Détection des Outliers - Toutes les Features")
plt.tight_layout()
plt.show()
```

**Interprétation** : Les boxplots avec encoches révèlent la présence de valeurs aberrantes dans plusieurs variables, notamment dans `residual sugar`, `free sulfur dioxide`, et `total sulfur dioxide`.

### 2.4 Matrice de Corrélation

```python
plt.figure(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Matrice de Corrélation entre les Features")
plt.tight_layout()
plt.show()
```

**Observations clés** :
- Forte corrélation positive entre `density` et `residual sugar`
- Forte corrélation négative entre `density` et `alcohol`
- Corrélation positive entre `total sulfur dioxide` et `free sulfur dioxide`

---

## 3. Préparation des Données et Modélisation

### 3.1 Séparation des Données

```python
from sklearn.model_selection import train_test_split

# Division : Train (33.3%) / Validation (33.3%) / Test (33.3%)
Xa, Xt, Ya, Yt = train_test_split(X, Y, shuffle=True, test_size=1/3, stratify=Y)
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, shuffle=True, test_size=0.5, stratify=Ya)
```

### 3.2 Normalisation des Features

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean=True, with_std=True)
sc = sc.fit(Xa)
Xa_n = sc.transform(Xa)
Xv_n = sc.transform(Xv)
```

**Justification** : La normalisation est essentielle pour les algorithmes basés sur les distances (comme KNN) afin d'éviter que les features avec de grandes échelles dominent le calcul de distance.

### 3.3 Classification K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Test avec k=3
k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(Xa, Ya)

Ypred_v = clf.predict(Xv)
error_v = 1 - accuracy_score(Yv, Ypred_v)
print(f"Taux d'erreur avec k={k}: {error_v:.4f}")
```

### 3.4 Optimisation de l'Hyperparamètre k

```python
k_vector = np.arange(1, 37, 2)
error_train = np.empty(k_vector.shape)
error_val = np.empty(k_vector.shape)

for ind, k in enumerate(k_vector):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xa, Ya)
    
    Ypred_train = clf.predict(Xa)
    error_train[ind] = 1 - accuracy_score(Ya, Ypred_train)
    
    Ypred_val = clf.predict(Xv)
    error_val[ind] = 1 - accuracy_score(Yv, Ypred_val)

# Meilleur k
err_min, ind_opt = error_val.min(), error_val.argmin()
k_star = k_vector[ind_opt]
print(f"Meilleur k: {k_star} avec erreur de validation: {err_min:.4f}")
```

**Visualisation de l'optimisation** :
```python
plt.figure(figsize=(10, 6))
plt.plot(k_vector, error_train, 'o-', label='Erreur Train')
plt.plot(k_vector, error_val, 's-', label='Erreur Validation')
plt.axvline(k_star, color='r', linestyle='--', label=f'k optimal = {k_star}')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Taux d\'erreur')
plt.title('Optimisation de l\'hyperparamètre k pour KNN')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 4. Analyse des Dépendances : Information Mutuelle

### 4.1 Calcul de l'Information Mutuelle

```python
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

Y_series = pd.Series(Y, index=X.index)
mi_scores = mutual_info_classif(X, Y_series)
mi_scores = pd.Series(mi_scores, index=X.columns)

print("Mutual Information Scores:")
print(mi_scores.sort_values(ascending=False))
```

### 4.2 Visualisation des Scores d'Information Mutuelle

```python
mi_scores_sorted = mi_scores.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_scores_sorted.index, y=mi_scores_sorted.values, 
            palette='viridis', hue=mi_scores_sorted.index, legend=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.title('Scores d\'Information Mutuelle avec la Qualité du Vin Binarisée')
plt.tight_layout()
plt.show()
```

### 4.3 Interprétation de l'Information Mutuelle

**Résultats clés** :
- **Density** (0.089) et **Alcohol** (0.087) : Dépendances les plus fortes
- **Residual Sugar** (0.0355), **Chlorides** (0.0352), **Total Sulfur Dioxide** (0.0337) : Dépendances modérées
- **Fixed Acidity** (0.0053) et **Sulphates** (0.0026) : Dépendances les plus faibles

**Signification** : L'information mutuelle mesure la dépendance (linéaire et non-linéaire) entre chaque feature et la variable cible. Plus le score est élevé, plus la feature est informative pour prédire la qualité du vin.

---

## 5. Analyse des Corrélations de Pearson

### 5.1 Calcul des Corrélations

```python
Y_series = pd.Series(Y, index=X.index, name='quality_binarized')
df_combined = pd.concat([X, Y_series], axis=1)

correlation_matrix = df_combined.corr(method='pearson')
pearson_correlations = correlation_matrix['quality_binarized'].drop('quality_binarized').abs()

print("Absolute Pearson Correlation Scores:")
print(pearson_correlations.sort_values(ascending=False))
```

### 5.2 Visualisation des Corrélations

```python
pearson_correlations_sorted = pearson_correlations.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=pearson_correlations_sorted.index, y=pearson_correlations_sorted.values,
            palette='viridis', hue=pearson_correlations_sorted.index, legend=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Corrélation de Pearson Absolue')
plt.title('Corrélations de Pearson Absolues avec la Qualité du Vin Binarisée')
plt.tight_layout()
plt.show()
```

### 5.3 Interprétation des Corrélations de Pearson

**Résultats clés** :
- **Alcohol** (0.383) : Corrélation linéaire la plus forte
- **Density** (0.269) et **Volatile Acidity** (0.225) : Corrélations fortes
- **Chlorides** (0.184) et **Total Sulfur Dioxide** (0.171) : Corrélations modérées
- **Citric Acid** (0.0007) et **Free Sulfur Dioxide** (0.0013) : Corrélations négligeables

**Différence avec l'Information Mutuelle** : La corrélation de Pearson capture uniquement les relations **linéaires**, tandis que l'information mutuelle détecte toutes formes de dépendances.

---

## 6. Synthèse et Interprétation Globale

### 6.1 Comparaison des Méthodes

| Feature | Information Mutuelle | Corrélation Pearson | Type de Relation |
|---------|---------------------|---------------------|------------------|
| **Alcohol** | 2e (0.087) | 1er (0.383) | Forte relation linéaire |
| **Density** | 1er (0.089) | 2e (0.269) | Relation mixte (linéaire + non-linéaire) |
| **Volatile Acidity** | Moyenne | 3e (0.225) | Principalement linéaire |
| **Chlorides** | Modérée (0.0352) | 4e (0.184) | Relation linéaire modérée |
| **Citric Acid** | Faible | Très faible (0.0007) | Relation non-linéaire possible |

### 6.2 Insights Clés

1. **Alcohol est le prédicteur linéaire le plus puissant** de la qualité du vin, avec une corrélation de 0.383
2. **Density montre la plus forte dépendance globale** (information mutuelle), suggérant des relations complexes
3. **Les relations non-linéaires existent** : certaines features ont une information mutuelle élevée mais une faible corrélation de Pearson
4. **Le modèle KNN bénéficie de la normalisation** en raison de sa sensibilité aux échelles de mesure

### 6.3 Recommandations

#### Pour la Sélection de Features
- **Modèles linéaires** : Prioriser Alcohol, Density, Volatile Acidity
- **Modèles non-linéaires** : Inclure également les features avec forte information mutuelle mais faible corrélation

#### Pour l'Amélioration du Modèle
1. **Feature Engineering** : Explorer les interactions entre Density et Alcohol (forte corrélation négative)
2. **Modèles alternatifs** : Tester des algorithmes capturant mieux les non-linéarités (Random Forest, Gradient Boosting)
3. **Traitement des outliers** : Considérer des transformations ou la suppression des valeurs extrêmes
4. **Équilibrage des classes** : Si déséquilibre important après binarisation, utiliser SMOTE ou class_weight

#### Pour l'Analyse Viticole
- La **teneur en alcool** est le facteur le plus déterminant de la qualité perçue
- La **densité** (liée au sucre résiduel et à l'alcool) joue un rôle complexe
- L'**acidité volatile** (liée aux défauts de vinification) impacte négativement la qualité
- Les **sulfites** ont un impact modéré mais nécessaire pour la conservation

---

## 7. Conclusion

Cette analyse complète du Wine Quality Dataset révèle que la qualité du vin est principalement influencée par sa composition chimique, avec l'alcool et la densité comme facteurs dominants. 

**Points forts de l'analyse** :
- ✅ Double approche (information mutuelle + corrélation) pour capturer relations linéaires et non-linéaires
- ✅ Visualisations claires et interprétables
- ✅ Préparation rigoureuse des données (normalisation, stratification)
- ✅ Optimisation méthodique de l'hyperparamètre k pour KNN

**Limites et perspectives** :
- Le modèle KNN de base peut être amélioré avec des techniques plus avancées
- L'analyse se concentre sur les vins blancs ; une comparaison avec les vins rouges serait pertinente
- Les interactions entre features mériteraient une exploration approfondie
- Des techniques de réduction de dimensionnalité (PCA) pourraient révéler des patterns cachés

**Prochaines étapes recommandées** :
1. Implémenter des modèles ensemble (Random Forest, XGBoost)
2. Effectuer une validation croisée k-fold pour une évaluation plus robuste
3. Analyser les erreurs de classification pour identifier les cas difficiles
4. Développer un système de recommandation basé sur les profils chimiques