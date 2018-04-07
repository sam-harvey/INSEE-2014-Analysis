
# coding: utf-8

# Une analyse du nombre d’établissements par secteur d’activité et par taille en 2014
# 
# https://www.insee.fr/fr/statistiques/1893274
# 
# https://www.kaggle.com/etiennelq/french-employment-by-town

# In[190]:


#chargement des bibliothèques logicielles
import pandas as pd
import numpy as np
import numbers

import re

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


# In[2]:


get_ipython().magic('qtconsole')


# Data Preparation

# In[14]:


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
        dep_index.append(i)
        
print('non-numeric departements:' + str(counter))

#Check the total column adds up
np.sum(df_tranche['E14TST'] == df_tranche.iloc[:,5:14].       apply(np.sum, axis = 1)) == df_tranche.shape[0]


# In[80]:


#Check geo data 
df_geo.dtypes

#Check for missing values
df_geo.head(5)

test = np.repeat(False, df_geo.shape[0])

#Across all columns
for col in df_geo.columns:
    #Find the rows with None records
    col_test = df_geo[col] == None
    test = test | col_test
    
    #Find rows with NaN records
    if isinstance(df_geo[col][0], numbers.Number):
        col_test2 = np.isnan(df_geo[col])
        test = test | col_test2

#Looks like the problem is missing long / lat / eloignement
df_geo[test]

#Convert to correct data types

#Matseille has three post codes delimited by whitespace
df_geo[df_geo.codes_postaux == '13001 13004 13012 ']

#Testing for castability
def num_castable(x):
    try:
        return(isinstance(float(x), float))
    except:
        return(False)

#So this is actually an array column delimited by whitespace
df_geo[np.logical_not(df_geo.codes_postaux.apply(num_castable))]

#Split and convert to float
#Need length for trailing white space
df_geo.codes_postaux = df_geo.codes_postaux.apply(lambda x: ([float(i) for i in x.split(" ") if len(i) > 0]))

#Do we need this?
df_geo.code_postale_max = df_geo.codes_postaux.apply(max)

#This has strings in it
df_geo[df_geo.numéro_département == '2A']

#Finally something that works
df_geo.code_insee.astype('int')

#Longitude
# ValueError: could not convert string to float: '3,28'
# df_geo.longitude.astype("float")

#They've mixed up French and English decimal notation, so it has loaded longitude as str
#Replace commas and convert to numeric if necessary
def decimal_translate(x):
    try:
        return(float(x))
    #Cases where it can't cast
    except:
        #French decimal
        x = re.sub(",", ".", x)
        #For where they are using - as null values
        if x == '-':
            x = float("NaN")
        return(float(x))
        
df_geo.longitude = df_geo.longitude.apply(decimal_translate)

#There are some with latitude but not longitude
df_geo[np.isnan(df_geo.latitude) * np.logical_not((np.isnan(df_geo.longitude)))]


# In[3]:


#
df_tranche.head(5)
df_tranche.describe()

df_salaire.head(5)
df_salaire.describe()

df_population.head(5)
df_population.describe()

df_geo.head(5)
df_geo.describe()

