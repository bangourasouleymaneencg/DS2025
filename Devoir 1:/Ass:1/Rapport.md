# Rapport d'Analyse sur la consommation électrique
## Consommation d'électricité des ménages individuels
### BANGOURA SOULEYMANE
### N° A : 22007304
### GROUPE 1 CAC
---
### <img src="s.jpeg" style="height:464px;margin-right:432px"/>	

<img src="SETTAT.png" style="height:464px;margin-right:432px"/>	


## Introduction au Dataset "Individual Household Electric Power Consumption"

### **Origine et Créateurs**

Ce dataset a été créé par Georges Hebrail, chercheur senior à EDF R&D à Clamart, France, et Alice Berard, étudiante en Master of Engineering à TELECOM ParisTech en stage à EDF R&D. Le jeu de données a été publié en 2006 et mis à disposition dans le UCI Machine Learning Repository.

### **Période et Localisation de Collecte**

Les données ont été collectées dans une maison située à Sceaux (à 7 km de Paris, France) entre décembre 2006 et novembre 2010, couvrant une période de 47 mois. Le dataset contient 2 075 259 mesures avec un taux d'échantillonnage d'une minute.

### **Méthode de Collecte**

Les mesures ont été effectuées par **EDF Energy** (Électricité de France), le principal fournisseur d'électricité français. La collecte s'est faite de manière automatisée avec des capteurs installés dans le foyer, enregistrant différentes grandeurs électriques minute par minute pendant près de 4 ans. Les mesures incluent à la fois la consommation globale et des sous-comptages pour des circuits spécifiques du logement.

### **Nature de la Population**

Il s'agit d'un **foyer résidentiel unique** situé dans une zone périurbaine proche de Paris. Le dataset représente donc une étude de cas approfondie d'un ménage français type sur une longue période, permettant d'observer les patterns de consommation quotidiens, hebdomadaires et saisonniers.

!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
individual_household_electric_power_consumption = fetch_ucirepo(id=235) 
  
# data (as pandas dataframes) 
X = individual_household_electric_power_consumption.data.features 
y = individual_household_electric_power_consumption.data.targets 
  
# metadata 
print(individual_household_electric_power_consumption.metadata) 
  
# variable information 
print(individual_household_electric_power_consumption.variables) 
/usr/local/lib/python3.12/dist-packages/ucimlrepo/fetch.py:97: DtypeWarning: Columns (2,3,4,5,6,7) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(data_url)
{'uci_id': 235, 'name': 'Individual Household Electric Power Consumption', 'repository_url': 'https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption', 'data_url': 'https://archive.ics.uci.edu/static/public/235/data.csv', 'abstract': 'Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available.', 'area': 'Physics and Chemistry', 'tasks': ['Regression', 'Clustering'], 'characteristics': ['Multivariate', 'Time-Series'], 'num_instances': 2075259, 'num_features': 9, 'feature_types': ['Real'], 'demographics': [], 'target_col': None, 'index_col': None, 'has_missing_values': 'no', 'missing_values_symbol': None, 'year_of_dataset_creation': 2006, 'last_updated': 'Fri Mar 08 2024', 'dataset_doi': '10.24432/C58K54', 'creators': ['Georges Hebrail', 'Alice Berard'], 'intro_paper': None, 'additional_info': {'summary': 'This archive contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months).\r\nNotes: \r\n1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.\r\n2.The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007.', 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': '1.date: Date in format dd/mm/yyyy\r\n2.time: time in format hh:mm:ss\r\n3.global_active_power: household global minute-averaged active power (in kilowatt)\r\n4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)\r\n5.voltage: minute-averaged voltage (in volt)\r\n6.global_intensity: household global minute-averaged current intensity (in ampere)\r\n7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).\r\n8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.\r\n9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.', 'citation': None}}
                    name     role         type demographic description units  \
0                   Date  Feature         Date        None        None  None   
1                   Time  Feature  Categorical        None        None  None   
2    Global_active_power  Feature   Continuous        None        None  None   
3  Global_reactive_power  Feature   Continuous        None        None  None   
4                Voltage  Feature   Continuous        None        None  None   
5       Global_intensity  Feature   Continuous        None        None  None   
6         Sub_metering_1  Feature   Continuous        None        None  None   
7         Sub_metering_2  Feature   Continuous        None        None  None   
8         Sub_metering_3  Feature   Continuous        None        None  None   

  missing_values  
0             no  
1             no  
2             no  
3             no  
4             no  
5             no  
6             no  
7             no  
8             no  


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ANALYSE DE CORRÉLATION - CONSOMMATION ÉLECTRIQUE RÉSIDENTIELLE")
print("=" * 80)
================================================================================
ANALYSE DE CORRÉLATION - CONSOMMATION ÉLECTRIQUE RÉSIDENTIELLE
================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ANALYSE DE CORRÉLATION - CONSOMMATION ÉLECTRIQUE RÉSIDENTIELLE")
print("=" * 80)

# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

print("\n[1] CHARGEMENT DES DONNÉES...")

# Use the already loaded X dataframe from ucimlrepo
df = X.copy()

# Combine Date and Time into a single datetime column
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Drop original 'Date' and 'Time' columns
df = df.drop(columns=['Date', 'Time'])

# Identify numerical columns and convert them to numeric, coercing errors to NaN
numerical_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Reorder columns to have 'datetime' first
cols_order = ['datetime'] + numerical_cols
df = df[cols_order]

print(f"✓ Dimensions du dataset: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"✓ Période: {df['datetime'].min()} à {df['datetime'].max()}")

# Afficher les premières lignes
print("\n📊 Aperçu des données:")
print(df.head())

# Informations sur les colonnes
print("\n📋 Structure du dataset:")
print(df.info())

================================================================================
ANALYSE DE CORRÉLATION - CONSOMMATION ÉLECTRIQUE RÉSIDENTIELLE
================================================================================

[1] CHARGEMENT DES DONNÉES...
✓ Dimensions du dataset: 2,075,259 lignes × 8 colonnes
✓ Période: 2006-12-16 17:24:00 à 2010-11-26 21:02:00

📊 Aperçu des données:
             datetime  Global_active_power  Global_reactive_power  Voltage  \
0 2006-12-16 17:24:00                4.216                  0.418   234.84   
1 2006-12-16 17:25:00                5.360                  0.436   233.63   
2 2006-12-16 17:26:00                5.374                  0.498   233.29   
3 2006-12-16 17:27:00                5.388                  0.502   233.74   
4 2006-12-16 17:28:00                3.666                  0.528   235.68   

   Global_intensity  Sub_metering_1  Sub_metering_2  Sub_metering_3  
0              18.4             0.0             1.0            17.0  
1              23.0             0.0             1.0            16.0  
2              23.0             0.0             2.0            17.0  
3              23.0             0.0             1.0            17.0  
4              15.8             0.0             1.0            17.0  

📋 Structure du dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2075259 entries, 0 to 2075258
Data columns (total 8 columns):
 #   Column                 Dtype         
---  ------                 -----         
 0   datetime               datetime64[ns]
 1   Global_active_power    float64       
 2   Global_reactive_power  float64       
 3   Voltage                float64       
 4   Global_intensity       float64       
 5   Sub_metering_1         float64       
 6   Sub_metering_2         float64       
 7   Sub_metering_3         float64       
dtypes: datetime64[ns](1), float64(7)
memory usage: 126.7 MB
None

# 2. NETTOYAGE DES DONNÉES
# ============================================================================

print("\n[2] NETTOYAGE DES DONNÉES...")

# Statistiques sur les valeurs manquantes
print("\n🔍 Valeurs manquantes par colonne:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Manquantes': missing,
    'Pourcentage': missing_pct
})
print(missing_df[missing_df['Manquantes'] > 0])

# Convertir les colonnes en numérique
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprimer les lignes avec trop de valeurs manquantes
initial_rows = len(df)
df = df.dropna()
print(f"✓ Lignes supprimées: {initial_rows - len(df):,} ({((initial_rows - len(df)) / initial_rows * 100):.2f}%)")
print(f"✓ Lignes restantes: {len(df):,}")

[2] NETTOYAGE DES DONNÉES...

🔍 Valeurs manquantes par colonne:
                       Manquantes  Pourcentage
Global_active_power         25979     1.251844
Global_reactive_power       25979     1.251844
Voltage                     25979     1.251844
Global_intensity            25979     1.251844
Sub_metering_1              25979     1.251844
Sub_metering_2              25979     1.251844
Sub_metering_3              25979     1.251844
✓ Lignes supprimées: 25,979 (1.25%)
✓ Lignes restantes: 2,049,280

# 3. STATISTIQUES DESCRIPTIVES
# ============================================================================

print("\n[3] STATISTIQUES DESCRIPTIVES...")

stats_desc = df[numeric_cols].describe()
print("\n📈 Statistiques descriptives:")
print(stats_desc.round(3))

[3] STATISTIQUES DESCRIPTIVES...

📈 Statistiques descriptives:
       Global_active_power  Global_reactive_power     Voltage  \
count          2049280.000            2049280.000  2049280.00   
mean                 1.092                  0.124      240.84   
std                  1.057                  0.113        3.24   
min                  0.076                  0.000      223.20   
25%                  0.308                  0.048      238.99   
50%                  0.602                  0.100      241.01   
75%                  1.528                  0.194      242.89   
max                 11.122                  1.390      254.15   

       Global_intensity  Sub_metering_1  Sub_metering_2  Sub_metering_3  
count       2049280.000     2049280.000     2049280.000     2049280.000  
mean              4.628           1.122           1.299           6.458  
std               4.444           6.153           5.822           8.437  
min               0.200           0.000           0.000           0.000  
25%               1.400           0.000           0.000           0.000  
50%               2.600           0.000           0.000           1.000  
75%               6.400           0.000           1.000          17.000  
max              48.400          88.000          80.000          31.000  

# 4. MATRICE DE CORRÉLATION COMPLÈTE
# ============================================================================

print("\n[4] CALCUL DE LA MATRICE DE CORRÉLATION...")

# Calculer la matrice de corrélation de Pearson
correlation_matrix = df[numeric_cols].corr(method='pearson')

print("\n🔢 Matrice de corrélation (Pearson):")
print(correlation_matrix.round(3))

# Visualisation: Heatmap de corrélation
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1, 
            vmax=1)
plt.title('Matrice de Corrélation - Consommation Électrique\n', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Heatmap sauvegardée: correlation_heatmap.png")
plt.show()

[4] CALCUL DE LA MATRICE DE CORRÉLATION...

🔢 Matrice de corrélation (Pearson):
                       Global_active_power  Global_reactive_power  Voltage  \
Global_active_power                  1.000                  0.247   -0.400   
Global_reactive_power                0.247                  1.000   -0.112   
Voltage                             -0.400                 -0.112    1.000   
Global_intensity                     0.999                  0.266   -0.411   
Sub_metering_1                       0.484                  0.123   -0.196   
Sub_metering_2                       0.435                  0.139   -0.167   
Sub_metering_3                       0.639                  0.090   -0.268   

                       Global_intensity  Sub_metering_1  Sub_metering_2  \
Global_active_power               0.999           0.484           0.435   
Global_reactive_power             0.266           0.123           0.139   
Voltage                          -0.411          -0.196          -0.167   
Global_intensity                  1.000           0.489           0.440   
Sub_metering_1                    0.489           1.000           0.055   
Sub_metering_2                    0.440           0.055           1.000   
Sub_metering_3                    0.627           0.103           0.081   

                       Sub_metering_3  
Global_active_power             0.639  
Global_reactive_power           0.090  
Voltage                        -0.268  
Global_intensity                0.627  
Sub_metering_1                  0.103  
Sub_metering_2                  0.081  
Sub_metering_3                  1.000  
✓ Heatmap sauvegardée: correlation_heatmap.png
