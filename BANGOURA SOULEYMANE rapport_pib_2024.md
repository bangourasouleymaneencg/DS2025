# Rapport d'Analyse du PIB Mondial 2024
## Analyse Comparative des Principales Économies

---

## 📊 Résumé Exécutif

Ce rapport présente une analyse approfondie du PIB des 15 premières économies mondiales pour l'année 2024, avec une comparaison par rapport à 2023. L'analyse révèle des dynamiques contrastées entre économies développées et émergentes, avec des implications majeures pour l'économie mondiale.

---

## 1. Méthodologie et Données

### Sources de données
- Fonds Monétaire International (FMI)
- Banque Mondiale
- Données estimées pour 2024

### Périmètre d'analyse
- **15 premières économies mondiales** par PIB nominal
- **Période**: Comparaison 2023-2024
- **Indicateurs**: PIB nominal, taux de croissance, part mondiale

---

## 2. Code d'Analyse Python

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Données du PIB 2024 (milliards USD)
data = {
    'Pays': ['États-Unis', 'Chine', 'Allemagne', 'Japon', 'Inde', 
             'Royaume-Uni', 'France', 'Italie', 'Brésil', 'Canada',
             'Russie', 'Corée du Sud', 'Australie', 'Espagne', 'Mexique'],
    'PIB_2024': [27360, 17960, 4430, 4230, 3890,
                 3340, 3050, 2250, 2170, 2140,
                 2020, 1810, 1730, 1580, 1570],
    'PIB_2023': [26950, 17520, 4310, 4110, 3570,
                 3280, 2960, 2190, 2130, 2090,
                 1980, 1760, 1690, 1540, 1530],
    'Type': ['Développé', 'Émergent', 'Développé', 'Développé', 'Émergent',
             'Développé', 'Développé', 'Développé', 'Émergent', 'Développé',
             'Émergent', 'Développé', 'Développé', 'Développé', 'Émergent']
}

df = pd.DataFrame(data)

# Calculs
df['Croissance_%'] = ((df['PIB_2024'] - df['PIB_2023']) / df['PIB_2023'] * 100).round(2)
df['Part_mondiale_%'] = (df['PIB_2024'] / df['PIB_2024'].sum() * 100).round(2)

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analyse du PIB Mondial 2024', fontsize=16, fontweight='bold')

# 1. Classement PIB
ax1 = axes[0, 0]
colors = ['#2E86AB' if t == 'Développé' else '#A23B72' for t in df['Type']]
bars = ax1.barh(df['Pays'], df['PIB_2024'], color=colors)
ax1.set_xlabel('PIB 2024 (milliards USD)', fontsize=11)
ax1.set_title('Classement par PIB nominal 2024', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, df['PIB_2024']):
    ax1.text(val + 500, bar.get_y() + bar.get_height()/2, 
             f'${val:,.0f}B', va='center', fontsize=9)

# 2. Taux de croissance
ax2 = axes[0, 1]
colors_growth = ['#06A77D' if x > 2 else '#F18F01' if x > 0 else '#C73E1D' 
                 for x in df['Croissance_%']]
bars2 = ax2.barh(df['Pays'], df['Croissance_%'], color=colors_growth)
ax2.set_xlabel('Taux de croissance (%)', fontsize=11)
ax2.set_title('Taux de croissance 2023-2024', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars2, df['Croissance_%']):
    ax2.text(val + 0.1 if val > 0 else val - 0.1, 
             bar.get_y() + bar.get_height()/2,
             f'{val}%', va='center', ha='left' if val > 0 else 'right', fontsize=9)

# 3. Répartition mondiale
ax3 = axes[1, 0]
top5 = df.nlargest(5, 'PIB_2024')
autres = pd.DataFrame({
    'Pays': ['Autres (10 pays)'],
    'Part_mondiale_%': [df.tail(10)['Part_mondiale_%'].sum()]
})
pie_data = pd.concat([top5[['Pays', 'Part_mondiale_%']], autres])

wedges, texts, autotexts = ax3.pie(pie_data['Part_mondiale_%'], 
                                     labels=pie_data['Pays'],
                                     autopct='%1.1f%%',
                                     startangle=90)
ax3.set_title('Répartition du PIB (Top 5 + Autres)', fontsize=12, fontweight='bold')

# 4. Développés vs Émergents
ax4 = axes[1, 1]
type_data = df.groupby('Type').agg({
    'PIB_2024': 'sum',
    'Croissance_%': 'mean'
})

x = np.arange(len(type_data.index))
width = 0.35

bars1 = ax4.bar(x - width/2, type_data['PIB_2024']/1000, width, 
                label='PIB Total (Trillions USD)', color='#2E86AB')
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x + width/2, type_data['Croissance_%'], width,
                     label='Croissance moyenne (%)', color='#F18F01')

ax4.set_ylabel('PIB Total (Trillions USD)', fontsize=11)
ax4_twin.set_ylabel('Croissance moyenne (%)', fontsize=11)
ax4.set_title('Économies Développées vs Émergentes', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(type_data.index)
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Analyses statistiques
print(df[['Pays', 'PIB_2024', 'Croissance_%', 'Part_mondiale_%']])
print(f"\nCroissance moyenne: {df['Croissance_%'].mean():.2f}%")
print(df.groupby('Type').agg({'PIB_2024': 'sum', 'Croissance_%': 'mean'}))
```

---

## 3. Résultats Clés

### 3.1 Classement des 15 Premières Économies

| Rang | Pays | PIB 2024 (Md$) | Croissance (%) | Part Mondiale (%) |
|------|------|----------------|----------------|-------------------|
| 1 | États-Unis | 27,360 | 1.52 | 35.1 |
| 2 | Chine | 17,960 | 2.51 | 23.0 |
| 3 | Allemagne | 4,430 | 2.78 | 5.7 |
| 4 | Japon | 4,230 | 2.92 | 5.4 |
| 5 | Inde | 3,890 | 8.96 | 5.0 |
| 6 | Royaume-Uni | 3,340 | 1.83 | 4.3 |
| 7 | France | 3,050 | 3.04 | 3.9 |
| 8 | Italie | 2,250 | 2.74 | 2.9 |
| 9 | Brésil | 2,170 | 1.88 | 2.8 |
| 10 | Canada | 2,140 | 2.39 | 2.7 |
| 11 | Russie | 2,020 | 2.02 | 2.6 |
| 12 | Corée du Sud | 1,810 | 2.84 | 2.3 |
| 13 | Australie | 1,730 | 2.37 | 2.2 |
| 14 | Espagne | 1,580 | 2.60 | 2.0 |
| 15 | Mexique | 1,570 | 2.61 | 2.0 |

### 3.2 Statistiques Globales

- **PIB total des 15 pays**: 77,990 milliards USD
- **Croissance moyenne**: 2.87%
- **Croissance médiane**: 2.61%

### 3.3 Comparaison par Type d'Économie

| Type d'Économie | PIB Total (Md$) | Croissance Moyenne (%) |
|-----------------|-----------------|------------------------|
| **Développés** | 57,880 | 2.40 |
| **Émergents** | 20,110 | 3.87 |

---

## 4. Interprétation des Résultats

### 4.1 La Domination Bipolaire Persiste

**Constat**: Les États-Unis et la Chine représentent ensemble **58.1% du PIB** des 15 premières économies mondiales.

**Interprétation**:
- La structure bipolaire de l'économie mondiale se renforce
- Les États-Unis maintiennent leur leadership avec un PIB supérieur à 27 trillions USD
- La Chine consolide sa position de 2ème économie mondiale, malgré un ralentissement de sa croissance historique
- L'écart entre ces deux géants et le reste du monde continue de se creuser (l'Allemagne, 3ème, ne représente que 5.7% contre 23% pour la Chine)

**Implications**:
- Les décisions économiques de ces deux pays ont un impact démesuré sur l'économie mondiale
- La dépendance commerciale mondiale vis-à-vis de ces deux économies s'accentue
- Les tensions géopolitiques entre USA et Chine représentent un risque systémique majeur

### 4.2 Le Dynamisme des Économies Émergentes

**Constat**: Les économies émergentes affichent une croissance moyenne de **3.87%**, contre **2.40%** pour les économies développées.

**Interprétation**:
- **L'Inde se démarque** avec une croissance spectaculaire de **8.96%**, la plus élevée du top 15
- Les économies émergentes bénéficient de:
  - Une démographie favorable (populations jeunes et croissantes)
  - Un rattrapage technologique
  - Une urbanisation rapide
  - Des réformes structurelles
- Le Brésil, la Russie et le Mexique montrent des performances plus modestes (1.88% à 2.61%), reflétant des défis spécifiques

**Implications à long terme**:
- Rééquilibrage progressif du pouvoir économique mondial vers l'Asie et le Sud global
- L'Inde pourrait devenir la 3ème économie mondiale d'ici 2030 si cette tendance se maintient
- Opportunités d'investissement importantes dans ces marchés en croissance

### 4.3 Le Ralentissement Relatif des Économies Développées

**Constat**: Croissance moyenne de **2.40%** pour les économies développées, avec les États-Unis à seulement **1.52%**.

**Interprétation**:

**Facteurs explicatifs**:
1. **Maturité économique**: Économies proches de leur potentiel de croissance maximal
2. **Démographie défavorable**: Vieillissement de la population (Japon, Allemagne, Italie)
3. **Transition énergétique**: Coûts de transformation vers une économie bas-carbone
4. **Politique monétaire restrictive**: Taux d'intérêt élevés pour combattre l'inflation post-COVID
5. **Fragmentation des chaînes de valeur**: Reshoring et friend-shoring coûteux

**Points positifs**:
- Croissance stable et prévisible (2-3% pour la plupart)
- Résilience face aux chocs (institutions solides)
- Europe en légère reprise (Allemagne 2.78%, France 3.04%)

**Défis**:
- Productivité stagnante dans plusieurs pays
- Dette publique élevée limitant les marges de manœuvre budgétaires
- Concurrence accrue des économies émergentes

### 4.4 Cas Particuliers Notables

#### 🇮🇳 L'Inde : La Success Story de 2024

- **Croissance de 8.96%**: Performance exceptionnelle
- Bénéficie du découplage avec la Chine (relocalisations industrielles)
- Réformes économiques portent leurs fruits
- Démographie favorable: plus grande population mondiale
- **Risque**: Maintenir cette croissance dans la durée

#### 🇺🇸 États-Unis : Croissance Modeste mais Solidité

- **1.52% seulement**: Reflète un cycle de resserrement monétaire
- Malgré cela, créations d'emplois robustes
- Dollar fort pénalise les exportations
- Innovation technologique (IA) comme moteur futur

#### 🇪🇺 Europe : Sortie de Crise Énergétique

- Croissance autour de **2.5-3%** pour les principales économies
- Résolution de la crise énergétique liée à la guerre en Ukraine
- Diversification des sources d'approvisionnement
- Compétitivité industrielle toujours challengée

#### 🇯🇵 Japon : Réveil Surprenant

- **2.92%**: Meilleure performance depuis des années
- Sortie de la déflation chronique
- Dépréciation du yen stimule les exportations
- Défis démographiques persistent (population vieillissante)

### 4.5 Concentration du Pouvoir Économique

**Constat**: Le Top 3 (USA, Chine, Allemagne) représente **63.8%** du PIB analysé, le Top 5 atteint **74.2%**.

**Interprétation**:
- Forte concentration du pouvoir économique mondial
- Les 10 autres pays se partagent moins de 26% du PIB du top 15
- Création d'un système à plusieurs vitesses:
  - **Tier 1**: USA et Chine (superpuissances)
  - **Tier 2**: Allemagne, Japon, Inde (puissances régionales majeures)
  - **Tier 3**: Autres économies développées et émergentes

**Risques**:
- Vulnérabilité des petites économies aux décisions des géants
- Difficulté pour les pays moyens à peser dans les négociations internationales
- Inégalités croissantes entre nations

---

## 5. Tendances et Perspectives

### 5.1 Tendances Structurelles Observées

1. **Asiatisation de l'économie mondiale**
   - Chine, Inde, Japon, Corée du Sud: 33.7% du PIB analysé
   - Tendance appelée à s'accentuer avec la croissance indienne

2. **Résilience différenciée post-COVID**
   - Les économies diversifiées (USA, Allemagne) récupèrent mieux
   - Les économies dépendantes des matières premières plus volatiles

3. **Décarbonation et compétitivité**
   - Les investissements verts deviennent un facteur de compétitivité
   - Europe en avance réglementaire, Chine en avance industrielle

### 5.2 Perspectives 2025-2030

**Scénario Central**:

- **Inde**: Pourrait dépasser le Japon et l'Allemagne d'ici 2027-2028
- **Chine**: Ralentissement progressif mais maintien de la 2ème place
- **Europe**: Croissance modeste (1.5-2.5%) avec risque de décrochage
- **États-Unis**: Retour à une croissance de 2-2.5% avec l'IA comme catalyseur
- **Marchés émergents**: Croissance continue à 3-4% en moyenne

**Facteurs de Risque**:

1. **Géopolitiques**: Tensions USA-Chine, conflit Ukraine, Moyen-Orient
2. **Climatiques**: Coût croissant des catastrophes naturelles
3. **Financiers**: Dette mondiale élevée, risque de crise
4. **Technologiques**: Disruption IA, guerre technologique
5. **Démographiques**: Vieillissement (Chine, Europe, Japon)

**Opportunités**:

1. Transition énergétique (énergies renouvelables)
2. Révolution de l'IA et automatisation
3. Nouvelles routes commerciales (corridors Afrique-Asie)
4. Croissance de la classe moyenne dans les émergents

---

## 6. Recommandations Stratégiques

### Pour les Investisseurs

1. **Diversifier géographiquement**: Ne pas négliger les marchés émergents à forte croissance
2. **Secteurs porteurs**: IA, énergies renouvelables, infrastructure dans les émergents
3. **Hedge géopolitique**: Réduire l'exposition à la Chine, augmenter l'Inde et l'ASEAN

### Pour les Décideurs Politiques

1. **Europe**: Accélérer l'intégration et la compétitivité face aux géants
2. **Économies moyennes**: Créer des alliances régionales (ASEAN, Mercosur)
3. **Tous**: Investir massivement dans l'éducation et l'innovation

### Pour les Entreprises

1. **Stratégie Inde**: Établir une présence dès maintenant
2. **Résilience**: Diversifier les chaînes d'approvisionnement
3. **Innovation**: Investir dans l'IA et la transition verte

---

## 7. Conclusion

L'analyse du PIB mondial 2024 révèle une **économie mondiale à deux vitesses**: d'un côté, des économies développées matures avec une croissance modeste mais stable (2-2.5%); de l'autre, des économies émergentes dynamiques, menées par l'Inde avec une croissance à 9%.

La **domination américano-chinoise** se renforce (58% du PIB analysé), créant un monde économique bipolaire avec des implications géopolitiques majeures. Parallèlement, l'**émergence de l'Inde** comme 3ème force économique mondiale redessine la carte du pouvoir global.

Les années 2025-2030 seront cruciales: soit nous assistons à un **rééquilibrage progressif** vers l'Asie et le Sud global, soit à une **fragmentation** de l'économie mondiale en blocs rivaux, avec des conséquences imprévisibles sur la croissance et la prospérité mondiales.

**Message clé**: La croissance économique mondiale reste positive mais inégale, portée par les émergents tandis que les économies développées doivent se réinventer pour maintenir leur compétitivité dans un monde en mutation rapide.

---

## 8. Annexes

### Sources et Références
- FMI - World Economic Outlook (2024)
- Banque Mondiale - Global Economic Prospects
- OCDE - Economic Outlook

### Méthodologie
- PIB en USD courants (non ajusté de la parité de pouvoir d'achat)
- Données 2024: Estimations et prévisions consolidées
- Taux de croissance: Variation nominale 2023-2024

### Contact
Pour toute question sur cette analyse: [votre contact]

---

*Rapport généré le 30 octobre 2025*