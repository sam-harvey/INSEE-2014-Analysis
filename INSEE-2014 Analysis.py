
# coding: utf-8

# Une analyse du nombre d’établissements par secteur d’activité et par taille en 2014
# 
# https://www.insee.fr/fr/statistiques/1893274
# 
# https://www.kaggle.com/etiennelq/french-employment-by-town

# In[41]:


#chargement des bibliothèques logicielles
import pandas as pd
import numpy as np
import json
import geojson

import os

import folium

#chargement des données
df_tranche = pd.read_csv('data/base_etablissement_par_tranche_effectif.csv')
df_salaire = pd.read_csv('data/net_salary_per_town_categories.csv')
df_population = pd.read_csv('data/population.csv')
df_geo = pd.read_csv('data/name_geographic_information.csv')

with open('./data/communes.geojson') as f:
    json_communes = json.load(f)

with open('./data/communes.geojson') as f:
    json_departements = json.load(f)

#Display all outputs from cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[95]:


get_ipython().magic('qtconsole')


# In[103]:


#Check tranche data
df_tranche.dtypes

# df_tranche.DEP.astype('int')
# df_tranche.CODGEO.astype('int')

#How many geo codes are not numeric?
counter = 0
geo_index = list()

for i in df_tranche.CODGEO:
    try: test = int(i)
    except:
        counter += 1
        geo_index.append(i)
        

print('non-numeric geo codes:' + str(counter))

#And Departements?
counter = 0
dep_index = list()

for i in df_tranche.DEP:
    try: test = int(i)
    except: 
        counter += 1
        dep_index = dep_index.append(i)

#These are the same?
dep_index == geo_index
        
print('non-numeric departements:' + str(counter))


#Check the total column adds up
np.sum(df_tranche['E14TST'] == df_tranche.iloc[:,5:14].       apply(np.sum, axis = 1)) == df_tranche.shape[0]


# In[125]:


#Check geo data 
df_geo.dtypes


# In[42]:


#
df_tranche.head(5)
df_tranche.describe()

df_salaire.head(5)
df_salaire.describe()

df_population.head(5)
df_population.describe()

df_geo.head(5)
df_geo.describe()

