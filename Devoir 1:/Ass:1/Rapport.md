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


```python
!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

```python
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

```python
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
  
