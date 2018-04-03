
# coding: utf-8

# Une analyse du nombre d’établissements par secteur d’activité et par taille en 2014
# 
# https://www.insee.fr/fr/statistiques/1893274

# In[9]:


#chargement des bibliothèques logicielles
import pandas as pd
import numpy as np

#chargement des données
df_tranche = pd.read_csv('data/base_etablissement_par_tranche_effectif.csv')
df_salaire = pd.read_csv('data/net_salary_per_town_categories.csv')
df_population = pd.read_csv('data/population.csv')
df_geo = pd.read_csv('data/name_geographic_information.csv')

#Display all outputs from cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[10]:


#
df_tranche.head(5)
df_tranche.describe()

df_salaire.head(5)
df_salaire.describe()

df_population.head(5)
df_population.describe()

df_geo.head(5)
df_geo.describe()

