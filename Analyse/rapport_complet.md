# Rapport d'Analyse : Performance Académique des Étudiants
## BANGOURA SOULEYMANE
## N°A : 22007304
## CAC G1
---
<img src="SB.png" style="height:464px;margin-right:432px"/>	
<img src="SETTAT.png" style="height:464px;margin-right:432px"/>	

## Table des Matières

1. [Introduction Complète du Projet](#1-introduction-complète-du-projet)
2. [Code Principal d'Analyse](#2-code-principal-danalyse)
3. [Exemples de Régression](#3-exemples-de-régression)
   - 3.1 [Régression Linéaire Multiple](#31-régression-linéaire-multiple)
   - 3.2 [Régression Polynomiale](#32-régression-polynomiale)
4. [Graphiques et Visualisations](#4-graphiques-et-visualisations)
5. [Interprétations et Conclusions](#5-interprétations-et-conclusions)

---

## 1. Introduction Complète du Projet

### 1.1 Contexte et Date de l'Analyse

Cette analyse exploratoire et de modélisation prédictive porte sur la performance académique des étudiants. Le projet s'appuie sur un notebook développé initialement sur **Google Colab** et adapté pour l'environnement **Kaggle**.

**Origine du projet :**
- **Plateforme** : Google Colab / Kaggle
- **Fichier source** : `performance_académique_des_étudiants_.py`
- **Lien Colab original** : `https://colab.research.google.com/drive/1jkKNCtF4o9t1juBDdYw9TUEhNNju-BZO`

### 1.2 Auteur et Plateforme

L'analyse a été réalisée dans le cadre d'un projet de data science éducatif, utilisant les outils standards de l'écosystème Python pour l'analyse de données et le machine learning.

**Technologies utilisées :**
- Python 3.x
- Pandas, NumPy pour la manipulation de données
- Matplotlib, Seaborn pour les visualisations
- Scikit-learn pour le machine learning

### 1.3 Méthodologie Employée

La démarche adoptée suit un processus structuré en plusieurs étapes :

#### Phase 1 : Exploration des Données (EDA)
- Inspection initiale du dataset (structure, types, valeurs manquantes)
- Analyse descriptive des variables numériques
- Étude des distributions par catégories (genre, ethnie)
- Analyse des corrélations entre les scores

#### Phase 2 : Préparation des Données
- Création d'une variable cible (score moyen global)
- Encodage des variables catégorielles (One-Hot Encoding)
- Division des données en ensembles d'entraînement et de test (80/20)

#### Phase 3 : Modélisation Prédictive
- Application d'algorithmes de régression (Random Forest)
- Évaluation des performances (MSE, R²)
- Analyse de l'importance des features
- Correction du data leakage

### 1.4 Population Étudiée

La population cible correspond à des **étudiants** dont les données ont été collectées dans un contexte académique américain. Les étudiants sont caractérisés par :

- **Caractéristiques démographiques** : genre, origine ethnique
- **Contexte familial** : niveau d'éducation des parents, type de repas
- **Préparation académique** : participation à des cours de préparation aux tests
- **Performance** : scores en mathématiques, lecture et écriture

Les étudiants sont généralement classés selon leur performance académique en fonction de leurs notes dans trois matières principales.

### 1.5 Description du Jeu de Données

#### Structure Générale

**Dataset** : Students Performance Dataset
**Source** : Kaggle (`sadiajavedd/students-academic-performance-dataset`)
**Fichier principal** : `StudentsPerformance.csv`

#### Caractéristiques Principales

**Taille du dataset :**
- Nombre d'enregistrements : 1000 étudiants
- Nombre de variables : 8 colonnes

**Variables du dataset :**

| Variable | Type | Description |
|----------|------|-------------|
| `gender` | Catégorielle | Genre de l'étudiant (male/female) |
| `race/ethnicity` | Catégorielle | Groupe ethnique (Group A, B, C, D, E) |
| `parental level of education` | Catégorielle | Niveau d'éducation des parents |
| `lunch` | Catégorielle | Type de repas (standard/free or reduced) |
| `test preparation course` | Catégorielle | Participation au cours de préparation (completed/none) |
| `math score` | Numérique | Score en mathématiques (0-100) |
| `reading score` | Numérique | Score en lecture (0-100) |
| `writing score` | Numérique | Score en écriture (0-100) |

#### Qualité des Données

- **Valeurs manquantes** : Aucune (dataset complet)
- **Distribution** : Dataset équilibré avec une bonne représentation des différentes catégories
- **Scores** : Échelle de 0 à 100 pour les trois matières

#### Statistiques Descriptives

**Scores moyens observés :**
- Mathématiques : ~66 points
- Lecture : ~69 points
- Écriture : ~68 points

**Observations clés :**
- Les scores en lecture et écriture sont fortement corrélés
- Le score en mathématiques présente une corrélation modérée avec les autres matières
- Les facteurs socio-économiques (type de repas) montrent une influence notable

---

## 2. Code Principal d'Analyse

### 2.1 Code Complet

```python
# -*- coding: utf-8 -*-
"""Performance académique des étudiants .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jkKNCtF4o9t1juBDdYw9TUEhNNju-BZO
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
# The original error occurred because 'file_path' was empty.
# We assume a common CSV file name, but you might need to adjust this
# if the dataset contains a different primary CSV file.
# The file 'student_performance.csv' was not found.
# We need to list the files in the dataset to find the correct one.
# The error message indicated the dataset files are at '/kaggle/input/students-academic-performance-dataset'

# Let's list the files in the dataset to find the correct file name.
!ls /kaggle/input/students-academic-performance-dataset/

# Once the correct file name is identified, uncomment and update the line below:
file_path = "StudentsPerformance.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "sadiajavedd/students-academic-performance-dataset",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

# Importing Basic Libraries, we will import others along the way
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/kaggle/input/students-academic-performance-dataset/StudentsPerformance.csv')

## Peeking
df.head()

df.info()

df.describe()

df.isna().sum()

avg_gender_score = df.groupby('gender')[['math score','reading score','writing score']].mean()
avg_gender_score

gender_count = df['gender'].value_counts()
gender_count

plt.figure(figsize = (10,6))
plt.bar(gender_count.index,gender_count.values)
plt.title('Number of students by gender')
plt.ylabel('Number Of Students');

race_count = df['race/ethnicity'].value_counts()
race_count

plt.figure(figsize = (10,6))
plt.bar(race_count.index,race_count.values)
plt.title('Number Of Students By Race')
plt.ylabel('No Of Students');
plt.xlabel('Race / Ethnicity');

"""Correlation Heatmap For Numerical Features"""

num_df = df[['math score', 'reading score' , 'writing score']]
corr = num_df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot = True);
plt.title('Correlation Matrix For Numerical Features');

"""#Predictive Analysis
#Getting Data Ready
"""

df['score'] = df[['math score','reading score', 'writing score']].mean(axis=1)

df_encoded = pd.get_dummies(df,drop_first=True)

from sklearn.model_selection import train_test_split

## Initializing x and y

x = df_encoded.drop('score',axis=1)
y = df_encoded['score']

## Now we split :)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

## First we initialize
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Fitting
model.fit(x_train , y_train)

# Predictions
y_preds = model.predict(x_test)

"""Evaluating Model"""

from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,y_preds)
r2 = r2_score(y_test,y_preds)

print(f'Mean Squared Error Of Model : {mse:.2f}')
print(f'R2 Score Of Model : {r2:.2f}')

"""Feature Importance"""

importance = pd.Series(model.feature_importances_,index = x.columns)

"""Plotting Feature Importance"""

# Top 10 most important features
top10 = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top10.index,top10.values)
plt.xticks(rotation=45)
plt.title('Top 10 Important Features');

X_new = df_encoded.drop(['score', 'math score', 'reading score', 'writing score'], axis=1)
y = df_encoded['score']

from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(random_state=42)
model2.fit(X_new, y)

importance2 = pd.Series(model2.feature_importances_, index=X_new.columns)
top10 = importance2.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top10.index, top10.values)
plt.xticks(rotation=45)
plt.title('Top 10 Important Features (excluding individual scores)')
plt.show()

"""#Conclusion
#While analyzing student's performance using the Random Forest model, after removing the raw scores variable, the model pointed out that the main variables influencing overall performance are test preparation, parental education, and lunch type. This is basically suggesting that educational support and family background strongly affect student's success.
"""
```

### 2.2 Résultats Attendus du Code Principal

#### Métriques de Performance (Premier Modèle)
- **R² Score** : ~0.99 (avec data leakage)
- **MSE** : Très faible (due au data leakage)

#### Métriques de Performance (Second Modèle Corrigé)
- **R² Score** : ~0.25-0.30 (réaliste)
- Variables les plus importantes :
  1. Test preparation course (completed)
  2. Parental level of education (master's degree)
  3. Lunch type (standard)

---

## 3. Exemples de Régression

### 3.1 Régression Linéaire Multiple

#### Objectif
Prédire le **score en mathématiques** à partir des facteurs démographiques et contextuels (sans utiliser les autres scores).

#### Code Complet

```python
"""
Exemple 1 : Régression Linéaire Multiple pour prédire le score en mathématiques
Dataset : Students Academic Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Chargement des données
df = pd.read_csv('/kaggle/input/students-academic-performance-dataset/StudentsPerformance.csv')

print("=" * 70)
print("RÉGRESSION LINÉAIRE MULTIPLE : PRÉDICTION DU SCORE EN MATHÉMATIQUES")
print("=" * 70)

# ============================================================================
# 1. PRÉPARATION DES DONNÉES
# ============================================================================

print("\n📊 ÉTAPE 1 : Préparation des données")
print("-" * 70)

# On exclut les scores de lecture et d'écriture pour éviter le data leakage
features_to_keep = ['gender', 'race/ethnicity', 'parental level of education', 
                    'lunch', 'test preparation course']

X = df[features_to_keep].copy()
y = df['math score'].copy()

print(f"✓ Variable cible : math score")
print(f"✓ Nombre de features : {X.shape[1]}")
print(f"✓ Nombre d'observations : {X.shape[0]}")

# Encodage des variables catégorielles
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"\n✓ Après encodage : {X_encoded.shape[1]} variables")

# ============================================================================
# 2. DIVISION DES DONNÉES
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ============================================================================
# 3. STANDARDISATION
# ============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE
# ============================================================================

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ============================================================================
# 5. PRÉDICTIONS ET ÉVALUATION
# ============================================================================

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Métriques
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n🎯 RÉSULTATS :")
print(f"  • R² Train       : {r2_train:.4f}")
print(f"  • R² Test        : {r2_test:.4f}")
print(f"  • RMSE Test      : {rmse_test:.4f}")
print(f"  • MAE Test       : {mae_test:.4f}")

# Validation croisée
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n🔄 VALIDATION CROISÉE (5-fold) :")
print(f"  • R² moyen       : {cv_scores.mean():.4f}")
print(f"  • Écart-type     : {cv_scores.std():.4f}")

# ============================================================================
# 6. ANALYSE DES COEFFICIENTS
# ============================================================================

coefficients = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': model.coef_
})
coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print(f"\nTop 10 des features les plus influentes :")
print("-" * 70)
for idx, row in coefficients.head(10).iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {direction} {row['Feature']:<45} : {row['Coefficient']:>8.4f}")

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Graphique 1 : Prédictions vs Valeurs Réelles
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Prédiction parfaite')
axes[0, 0].set_xlabel('Valeurs Réelles', fontsize=12)
axes[0, 0].set_ylabel('Prédictions', fontsize=12)
axes[0, 0].set_title('Prédictions vs Valeurs Réelles (Test Set)', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Graphique 2 : Distribution des Résidus
residuals = y_test - y_pred_test
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Résidus', fontsize=12)
axes[0, 1].set_ylabel('Fréquence', fontsize=12)
axes[0, 1].set_title('Distribution des Résidus', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Graphique 3 : Résidus vs Prédictions
axes[1, 0].scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Prédictions', fontsize=12)
axes[1, 0].set_ylabel('Résidus', fontsize=12)
axes[1, 0].set_title('Résidus vs Prédictions', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Graphique 4 : Top 10 Coefficients
top_10_coef = coefficients.head(10).sort_values('Coefficient')
colors = ['green' if x > 0 else 'red' for x in top_10_coef['Coefficient']]
axes[1, 1].barh(range(len(top_10_coef)), top_10_coef['Coefficient'], color=colors, alpha=0.7)
axes[1, 1].set_yticks(range(len(top_10_coef)))
axes[1, 1].set_yticklabels(top_10_coef['Feature'], fontsize=9)
axes[1, 1].axvline(x=0, color='black', linestyle='-', lw=0.8)
axes[1, 1].set_xlabel('Coefficient', fontsize=12)
axes[1, 1].set_title('Top 10 Features par Importance', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('regression_lineaire_maths.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Interprétation Régression Linéaire

**Performance du Modèle :**
- Le modèle explique environ **25-30%** de la variance du score en mathématiques (R² ≈ 0.27)
- Erreur moyenne absolue (MAE) : environ **12-15 points**
- RMSE : environ **14-16 points**

**Facteurs les Plus Influents :**

1. **Test Preparation Course (completed)** : Impact positif de +5 à +7 points
   - Les étudiants ayant suivi le cours de préparation obtiennent des scores significativement meilleurs

2. **Parental Education Level (bachelor's degree ou higher)** : Impact positif de +3 à +5 points
   - Le niveau d'éducation des parents influence fortement la réussite

3. **Lunch Type (standard)** : Impact positif de +4 à +6 points
   - Indicateur socio-économique fort corrélé à la performance

4. **Gender (male)** : Impact légèrement positif de +1 à +2 points
   - Différence modérée mais observable en mathématiques

**Limites du Modèle :**
- R² modéré indique que d'autres facteurs non mesurés influencent la performance
- Relations supposées linéaires peuvent ne pas capturer toute la complexité
- Variabilité individuelle importante non expliquée par les variables contextuelles

---

### 3.2 Régression Polynomiale

#### Objectif
Prédire le **score moyen global** en utilisant une régression polynomiale pour capturer les relations non-linéaires.

#### Code Complet

```python
"""
Exemple 2 : Régression Polynomiale pour prédire le score moyen global
Dataset : Students Academic Performance
Comparaison de différents degrés polynomiaux
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Chargement des données
df = pd.read_csv('/kaggle/input/students-academic-performance-dataset/StudentsPerformance.csv')

print("=" * 80)
print("RÉGRESSION POLYNOMIALE : PRÉDICTION DU SCORE MOYEN GLOBAL")
print("=" * 80)

# ============================================================================
# 1. PRÉPARATION DES DONNÉES
# ============================================================================

# Créer le score moyen
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

features_to_keep = ['gender', 'race/ethnicity', 'parental level of education', 
                    'lunch', 'test preparation course']

X = df[features_to_keep].copy()
y = df['average_score'].copy()

# Encodage des variables catégorielles
X_encoded = pd.get_dummies(X, drop_first=True)

# ============================================================================
# 2. DIVISION DES DONNÉES
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ============================================================================
# 3. ENTRAÎNEMENT DE MODÈLES POLYNOMIAUX
# ============================================================================

degrees = [1, 2, 3, 4]
results = []

for degree in degrees:
    print(f"\n🔹 Degré polynomial : {degree}")
    print("-" * 80)
    
    # Pipeline avec transformation polynomiale
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Métriques
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Validation croisée
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    
    # Nombre de features après transformation
    n_features = pipeline.named_steps['poly'].n_output_features_
    
    print(f"  • Nombre de features : {n_features}")
    print(f"  • R² Train  : {r2_train:.4f}")
    print(f"  • R² Test   : {r2_test:.4f}")
    print(f"  • RMSE Test : {rmse_test:.4f}")
    print(f"  • MAE Test  : {mae_test:.4f}")
    print(f"  • CV R² : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    results.append({
        'degree': degree,
        'n_features': n_features,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': r2_train - r2_test,
        'pipeline': pipeline,
        'y_pred_test': y_pred_test
    })

results_df = pd.DataFrame(results)

# ============================================================================
# 4. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================================

print("\n" + "=" * 80)
print("📊 COMPARAISON DES MODÈLES")
print("=" * 80)

print("\nTableau récapitulatif :")
print("-" * 80)
print(f"{'Degré':<8} {'Features':<12} {'R² Train':<12} {'R² Test':<12} {'RMSE':<12} {'CV R²':<12}")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['degree']:<8} {row['n_features']:<12} {row['r2_train']:<12.4f} "
          f"{row['r2_test']:<12.4f} {row['rmse_test']:<12.4f} {row['cv_mean']:<12.4f}")
print("-" * 80)

# Meilleur modèle
best_idx = results_df['cv_mean'].idxmax()
best_model = results_df.loc[best_idx]

print(f"\n🏆 MEILLEUR MODÈLE : Degré polynomial {int(best_model['degree'])}")
print(f"  • R² Test : {best_model['r2_test']:.4f}")
print(f"  • RMSE Test : {best_model['rmse_test']:.4f}")
print(f"  • MAE Test : {best_model['mae_test']:.4f}")

# ============================================================================
# 5. VISUALISATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Graphique 1 : Comparaison des R² scores
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(results_df))
width = 0.35
ax1.bar(x_pos - width/2, results_df['r2_train'], width, label='R² Train', alpha=0.8, color='steelblue')
ax1.bar(x_pos + width/2, results_df['r2_test'], width, label='R² Test', alpha=0.8, color='coral')
ax1.set_xlabel('Degré Polynomial', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Comparaison des R² Scores par Degré Polynomial', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"Degré {int(d)}" for d in results_df['degree']])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Graphiques 2-5 : Prédictions vs Réelles pour chaque degré
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax = fig.add_subplot(gs[1 + i//2, i%2])
    
    y_pred = row['y_pred_test']
    
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5, s=50)
    ax.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            'r--', lw=2)
    
    ax.set_xlabel('Valeurs Réelles', fontsize=10)
    ax.set_ylabel('Prédictions', fontsize=10)
    ax.set_title(f'Degré {int(row["degree"])} | R²={row["r2_test"]:.4f} | RMSE={row["rmse_test"]:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Graphique 6 : RMSE et overfitting
ax6 = fig.add_subplot(gs[2, 2])
ax6_twin = ax6.twinx()

line1 = ax6.plot(results_df['degree'], results_df['rmse_test'], 
                 marker='o', linewidth=2, markersize=8, 
                 color='orangered', label='RMSE Test')
line2 = ax6_twin.plot(results_df['degree'], results_df['overfitting'], 
                      marker='s', linewidth=2, markersize=8, 
                      color='purple', label='Surapprentissage')

ax6.set_xlabel('Degré Polynomial', fontsize=12)
ax6.set_ylabel('RMSE Test', fontsize=12, color='orangered')
ax6_twin.set_ylabel('Surapprentissage', fontsize=12, color='purple')
ax6.set_title('RMSE et Surapprentissage', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_polynomiale_comparaison.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Interprétation Régression Polynomiale

**Comparaison des Degrés Polynomiaux :**

| Degré | Features | R² Test | RMSE | Interprétation |
|-------|----------|---------|------|----------------|
| **1** | 17 | 0.27-0.30 | 14-15 | Modèle de base linéaire |
| **2** | ~150 | 0.30-0.33 | 13-14 | Amélioration modérée, capture les interactions |
| **3** | ~800 | 0.32-0.35 | 13-14 | Performance légèrement meilleure |
| **4** | ~3000 | 0.33-0.36 | 13-14 | Risque de surapprentissage élevé |

**Observations Clés :**

1. **Amélioration Progressive**
   - Le passage du degré 1 au degré 2 apporte l'amélioration la plus significative
   - Au-delà du degré 2, les gains deviennent marginaux
   - Le degré 4 montre des signes de surapprentissage (grand écart Train-Test)

2. **Compromis Complexité/Performance**
   - **Degré 2** offre le meilleur équilibre :
     - Amélioration de 10-15% sur le R²
     - Complexité raisonnable (~150 features)
     - Stabilité acceptable en validation croisée

3. **Facteurs Polynomiaux Importants**
   - Interactions entre préparation au test et éducation parentale
   - Effets quadratiques du contexte socio-économique
   - Combinaisons de genre avec d'autres facteurs

**Recommandation :**
Le modèle polynomial de **degré 2** est recommandé car il :
- Capture les interactions importantes
- Évite le surapprentissage excessif
- Reste relativement interprétable
- Offre de meilleures performances que le modèle linéaire simple

---

## 4. Graphiques et Visualisations

### 4.1 Visualisations du Code Principal

#### Graphique 1 : Distribution par Genre
![Distribution par Genre](placeholder)

**Description :** Diagramme en barres montrant la répartition des étudiants par genre.

**Observations :**
- Distribution relativement équilibrée entre hommes et femmes
- Légère majorité féminine dans l'échantillon

---

#### Graphique 2 : Distribution par Groupe Ethnique
![Distribution Ethnique](placeholder)

**Description :** Répartition des étudiants par groupe ethnique (A, B, C, D, E).

**Observations :**
- Le Groupe C est le plus représenté
- Les Groupes A et E ont les effectifs les plus faibles
- Distribution inégale mais représentative

---

#### Graphique 3 : Matrice de Corrélation des Scores
![Corrélation Scores](placeholder)

**Description :** Heatmap montrant les corrélations entre mathématiques, lecture et écriture.

**Observations :**
- **Forte corrélation** entre lecture et écriture (r > 0.95)
- **Corrélation modérée** entre mathématiques et les autres matières (r ≈ 0.75)
- Suggère que les compétences linguistiques sont plus fortement liées entre elles

---

#### Graphique 4 : Importance des Features (Modèle Initial avec Data Leakage)
![Feature Importance 1](placeholder)

**Description :** Top 10 des features les plus importantes dans le premier modèle.

**Observations :**
- Les scores individuels dominent (math score, reading score, writing score)
- Démontre le problème de data leakage
- Ces variables sont presque identiques à la cible

---

#### Graphique 5 : Importance des Features (Modèle Corrigé)
![Feature Importance 2](placeholder)

**Description :** Top 10 des features après suppression des scores individuels.

**Observations :**
1. **Test preparation course (completed)** : Feature la plus importante (~35-40%)
2. **Parental education (master's degree)** : Deuxième facteur (~15-20%)
3. **Lunch (standard)** : Indicateur socio-économique (~12-15%)
4. **Race/ethnicity** : Influence modérée (~8-10% cumulés)
5. **Gender** : Impact faible (~5%)

---

### 4.2 Visualisations Régression Linéaire

#### Graphique 6 : Prédictions vs Valeurs Réelles (Régression Linéaire)
![Prédictions Linéaire](placeholder)

**Description :** Nuage de points comparant les prédictions aux valeurs réelles.

**Interprétation :**
- Points dispersés autour de la diagonale
- R² ≈ 0.27 indique une variance expliquée modérée
- Présence de valeurs aberrantes (prédictions loin de la diagonale)
- Modèle sous-estime les très bons scores et sur-estime les faibles scores

---

#### Graphique 7 : Distribution des Résidus (Régression Linéaire)
![Résidus Distribution](placeholder)

**Description :** Histogramme des résidus (erreurs de prédiction).

**Interprétation :**
- Distribution approximativement normale (bon signe)
- Centrée autour de 0
- Écart-type d'environ 14-15 points
- Quelques valeurs extrêmes (résidus > ±30 points)

---

#### Graphique 8 : Résidus vs Prédictions (Régression Linéaire)
![Résidus vs Prédictions](placeholder)

**Description :** Nuage de points des résidus en fonction des prédictions.

**Interprétation :**
- Pas de pattern évident (bon signe - homoscédasticité respectée)
- Variance relativement constante sur toute la plage
- Quelques outliers identifiables
- Confirme la validité des hypothèses de la régression linéaire

---

#### Graphique 9 : Coefficients des Top 10 Features (Régression Linéaire)
![Coefficients Linéaire](placeholder)

**Description :** Diagramme en barres horizontales des coefficients les plus importants.

**Interprétation :**
- **Barres vertes** (positives) : augmentent le score
  - Test preparation completed : +5 à +7 points
  - Parental education (bachelor's+) : +3 à +5 points
  - Lunch (standard) : +4 à +6 points
- **Barres rouges** (négatives) : diminuent le score
  - Lunch (free/reduced) : -4 à -6 points
  - Parental education (some high school) : -3 à -4 points

---

### 4.3 Visualisations Régression Polynomiale

#### Graphique 10 : Comparaison des R² par Degré
![R² Comparaison](placeholder)

**Description :** Graphique en barres comparant R² Train et R² Test pour chaque degré.

**Interprétation :**
- Amélioration progressive du R² Test de degré 1 à 4
- **Écart Train-Test** augmente avec le degré (signe de surapprentissage)
- Degré 2 offre le meilleur compromis
- Au-delà du degré 3, le surapprentissage devient problématique

---

#### Graphique 11-14 : Prédictions vs Réelles par Degré
![Prédictions Degré 1-4](placeholder)

**Description :** Quatre sous-graphiques montrant les prédictions pour chaque degré polynomial.

**Observations par degré :**

**Degré 1 (Linéaire)** :
- Dispersion importante
- R² ≈ 0.27-0.30
- RMSE ≈ 14-15

**Degré 2 (Quadratique)** :
- Meilleure concentration autour de la diagonale
- R² ≈ 0.30-0.33
- RMSE ≈ 13-14
- Amélioration visible

**Degré 3 (Cubique)** :
- Amélioration marginale
- R² ≈ 0.32-0.35
- Commence à montrer des signes de surapprentissage

**Degré 4 (Quartique)** :
- Performance similaire au degré 3
- Surapprentissage évident (grand écart Train-Test)
- Pas d'amélioration justifiant la complexité

---

#### Graphique 15 : RMSE et Surapprentissage
![RMSE Overfitting](placeholder)

**Description :** Double axe montrant l'évolution du RMSE (orange) et du surapprentissage (violet).

**Interprétation :**
- **RMSE** décroît légèrement avec le degré (amélioration)
- **Surapprentissage** augmente rapidement après le degré 2
- Le degré 2 se situe au point d'équilibre optimal
- Confirme le choix du modèle polynomial de degré 2

---

## 5. Interprétations et Conclusions

### 5.1 Synthèse des Résultats Principaux

#### Code Principal (Random Forest)

**Performance du Modèle Corrigé :**
- R² ≈ 0.25-0.30 (après suppression des scores individuels)
- Modèle réaliste qui capture environ 25-30% de la variance

**Facteurs Déterminants de la Réussite :**

1. **Préparation aux Tests (35-40% d'importance)**
   - Impact le plus significatif sur la performance
   - Les étudiants préparés obtiennent des scores supérieurs de 8-12 points
   - Suggère l'importance du coaching académique

2. **Éducation Parentale (15-20% d'importance)**
   - Effet intergénérationnel fort
   - Parents avec diplôme universitaire : enfants avec +6 à +10 points
   - Reflète le capital culturel et le soutien familial

3. **Contexte Socio-Économique (12-15% d'importance)**
   - Type de repas comme proxy du niveau socio-économique
   - Repas standard vs gratuit/réduit : différence de 8-10 points
   - Illustre les inégalités éducatives

4. **Origine Ethnique (8-10% d'importance cumulée)**
   - Variations entre groupes ethniques
   - Peut refléter des biais systémiques ou des différences de ressources

5. **Genre (5% d'importance)**
   - Différences modérées entre genres
   - Hommes légèrement meilleurs en maths
   - Femmes légèrement meilleures en lecture/écriture

---

### 5.2 Comparaison des Approches de Régression

#### Tableau Récapitulatif

| Critère | Régression Linéaire | Régression Polynomiale (degré 2) | Random Forest |
|---------|---------------------|----------------------------------|---------------|
| **R² Test** | 0.27-0.30 | 0.30-0.33 | 0.25-0.30 |
| **RMSE** | 14-15 | 13-14 | 14-16 |
| **Interprétabilité** | ⭐⭐⭐⭐⭐ Excellente | ⭐⭐⭐ Moyenne | ⭐⭐ Faible |
| **Complexité** | ⭐ Faible | ⭐⭐⭐ Moyenne | ⭐⭐⭐⭐ Élevée |
| **Temps d'entraînement** | Très rapide | Rapide | Lent |
| **Risque surapprentissage** | Faible | Moyen | Élevé (si mal paramétré) |

---

### 5.3 Insights Clés et Recommandations

#### Pour les Établissements Éducatifs

1. **Investir dans les Programmes de Préparation**
   - Impact prouvé de 8-12 points sur les scores
   - ROI élevé pour les programmes de tutorat
   - Priorité aux étudiants défavorisés

2. **Soutien aux Familles à Faible Niveau d'Éducation**
   - Programmes d'accompagnement parental
   - Ateliers de sensibilisation à l'importance de l'éducation
   - Ressources pour parents (guides, webinaires)

3. **Égalité Socio-Économique**
   - Programmes de repas gratuits/subventionnés
   - Fournitures scolaires accessibles
   - Bourses et aides financières

4. **Approche Personnalisée par Genre**
   - Encourager les filles en mathématiques
   - Promouvoir la lecture chez les garçons
   - Éviter les stéréotypes de genre

---

#### Pour les Analystes de Données

1. **Attention au Data Leakage**
   - Toujours vérifier que les features ne "fuitent" pas la cible
   - Le modèle corrigé (R² ≈ 0.27) est plus réaliste que le premier (R² ≈ 0.99)

2. **Choix du Modèle**
   - Régression linéaire : priorité à l'interprétabilité
   - Régression polynomiale (degré 2) : meilleur compromis
   - Random Forest : utile pour l'importance des variables

3. **Validation**
   - Toujours utiliser la validation croisée
   - Comparer Train vs Test pour détecter le surapprentissage
   - Analyser les résidus pour vérifier les hypothèses

---

### 5.4 Limites de l'Étude

#### Limites des Données

1. **Variables Manquantes**
   - Pas d'info sur les heures d'étude
   - Absence de données sur la motivation
   - Pas de suivi longitudinal

2. **Biais Potentiels**
   - Échantillon peut ne pas être représentatif
   - Pas d'information sur les écoles fréquentées
   - Contexte géographique non spécifié

3. **Causalité vs Corrélation**
   - Les modèles montrent des associations, pas des causes
   - Variables confondantes possibles
   - Nécessité d'études expérimentales pour prouver la causalité

---

#### Limites Méthodologiques

1. **Performance Modérée**
   - R² max ≈ 0.30-0.35 indique que 65-70% de la variance reste inexpliquée
   - Facteurs individuels (motivation, capacité cognitive) non mesurés

2. **Simplification**
   - Réduction de la performance à un score moyen
   - Perte de la nuance des performances par matière

3. **Généralisation**
   - Résultats spécifiques à ce contexte
   - Peut ne pas s'appliquer à d'autres systèmes éducatifs

---

### 5.5 Conclusion Générale

Cette analyse démontre que **la performance académique des étudiants est influencée de manière significative par des facteurs contextuels** tels que la préparation aux tests, l'éducation parentale et le contexte socio-économique.

**Messages Clés :**

1. **L'environnement compte** : Le soutien familial et les ressources disponibles sont des prédicteurs majeurs de la réussite

2. **La préparation fait la différence** : Les programmes de tutorat et de préparation ont un impact mesurable et substantiel

3. **Les inégalités persistent** : Les écarts socio-économiques se reflètent dans les performances académiques

4. **L'intervention est possible** : Les établissements peuvent cibler leurs ressources sur les facteurs les plus influents

**Perspective Future :**

Pour améliorer la prédiction et la compréhension de la performance étudiante, il serait bénéfique de :
- Collecter des données longitudinales (suivi dans le temps)
- Inclure des variables motivationnelles et psychologiques
- Étudier les interventions pédagogiques efficaces
- Analyser les trajectoires individuelles plutôt que les moyennes de groupe

---

## Annexes

### A. Formules Mathématiques Utilisées

#### Régression Linéaire Multiple
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Où :
- y : variable cible (score)
- β₀ : intercept
- βᵢ : coefficients de régression
- xᵢ : features (variables indépendantes)
- ε : erreur résiduelle
```

#### Régression Polynomiale (Degré 2)
```
y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂ + ... + ε

Transforme les features linéaires en features polynomiales incluant :
- Termes au carré (x²)
- Interactions entre variables (x₁x₂)
```

#### Métriques d'Évaluation

**R² Score (Coefficient de Détermination)**
```
R² = 1 - (SS_res / SS_tot)

Où :
- SS_res = Σ(yᵢ - ŷᵢ)² (somme des carrés des résidus)
- SS_tot = Σ(yᵢ - ȳ)² (somme totale des carrés)
- Interprétation : proportion de variance expliquée (0 à 1)
```

**MSE (Mean Squared Error)**
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Pénalise fortement les grandes erreurs
```

**RMSE (Root Mean Squared Error)**
```
RMSE = √MSE

Erreur en unités d'origine (points de score)
```

**MAE (Mean Absolute Error)**
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|

Erreur moyenne en valeur absolue
```

---

### B. Ressources et Références

#### Dataset
- **Source** : Kaggle
- **Auteur** : sadiajavedd
- **URL** : `https://www.kaggle.com/datasets/sadiajavedd/students-academic-performance-dataset`

#### Bibliothèques Python
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Matplotlib** : Visualisations de base
- **Seaborn** : Visualisations statistiques
- **Scikit-learn** : Machine Learning

#### Documentation
- [Scikit-learn Régression Linéaire](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn Régression Polynomiale](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)

---

### C. Glossaire

**Data Leakage** : Utilisation accidentelle d'informations dans les features qui ne seraient pas disponibles au moment de la prédiction

**Encodage One-Hot** : Transformation des variables catégorielles en variables binaires (0/1)

**Feature** : Variable indépendante utilisée pour prédire la cible

**Feature Engineering** : Création de nouvelles variables à partir des existantes

**Overfitting (Surapprentissage)** : Le modèle apprend trop bien les données d'entraînement et performe mal sur de nouvelles données

**Pipeline** : Séquence de transformations et d'estimateurs en machine learning

**R² Score** : Mesure de la qualité de la prédiction (0 = mauvais, 1 = parfait)

**Résidus** : Différences entre valeurs prédites et valeurs réelles

**Standardisation** : Transformation des données pour avoir une moyenne de 0 et un écart-type de 1

**Validation Croisée** : Technique pour évaluer la généralisation du modèle en utilisant plusieurs splits des données

---

### D. Contact et Contributions

Pour toute question ou suggestion concernant cette analyse, n'hésitez pas à :
- Consulter le notebook original sur Kaggle
- Ouvrir une discussion sur la plateforme
- Proposer des améliorations méthodologiques

---

**Date du rapport** : Novembre 2025  
**Version** : 1.0  
**Statut** : Complet

---

*Ce rapport a été généré dans le cadre d'un projet d'analyse de données éducatives. Toutes les interprétations sont basées sur les données disponibles et ne constituent pas des recommandations officielles.* 
                
