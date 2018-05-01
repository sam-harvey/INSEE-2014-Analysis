
# coding: utf-8

# Une analyse du nombre d’établissements par secteur d’activité et par taille en 2014
# 
# https://www.insee.fr/fr/statistiques/1893274
# 
# https://www.kaggle.com/etiennelq/french-employment-by-town

# In[1]:


#chargement des bibliothèques logicielles
import pandas as pd
import numpy as np
import numbers

import re

import json
import geojson

import os

import folium

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import rgb2hex

from sklearn.cluster import KMeans

#chargement des données
df_tranche = pd.read_csv('data/base_etablissement_par_tranche_effectif.csv')
df_salaire = pd.read_csv('data/net_salary_per_town_categories.csv')
df_population = pd.read_csv('data/population.csv')
df_geo = pd.read_csv('data/name_geographic_information.csv')

with open('./data/communes.geojson') as f:
    json_communes = json.load(f)

with open('./data/departements.geojson') as f:
    json_departements = json.load(f)

#Display all outputs from cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


get_ipython().magic('qtconsole')


# Data Preparation

# base_etablissement_par_tranche_effectif.csv

# In[3]:


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


# name_geographic_information.csv

# In[4]:


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
#length condition deals with trailing white space
df_geo.codes_postaux = df_geo.codes_postaux.apply(lambda x: ([float(i) for i in x.split(" ") if len(i) > 0]))

#Do we need this?
df_geo['code_postale_max'] = df_geo.codes_postaux.apply(max)

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
        #For where they are using '-' as a null value
        if x == '-':
            x = float("NaN")
        return(float(x))
        
df_geo.longitude = df_geo.longitude.apply(decimal_translate)

#There are some with latitude but not longitude
df_geo[np.isnan(df_geo.latitude) & np.logical_not((np.isnan(df_geo.longitude)))]


# net_salary_per_town_categories.csv

# In[5]:


df_salaire.dtypes

#Unique ID is unique
df_salaire.CODGEO.nunique() == df_salaire.shape[0]
df_salaire.LIBGEO.nunique() == df_salaire.shape[0]

#Town names are not unique.
df_salaire.LIBGEO.value_counts()

df_salaire.describe()

#SNHMF1814 : mean net salary per hour for women between 18-25 years old
#Why does this have the highest max of all columns?

#The max is a tiny village. Is there one really high earner that has skewed this?
#This value is also reflected in SNHM1814
#Well there's not much to do about this
df_salaire[df_salaire.SNHMH1814 == max(df_salaire.SNHMH1814)]


# population.csv

# In[6]:


df_population.describe(include = 'all')

df_population.head(5)

#This is useless
df_population.NIVGEO.unique()

#Categorical Variables
df_population['SEXE'] = df_population['SEXE'].astype('category')
df_population['SEXE'].cat.categories = ["Male", "Female"]

# MOCO : cohabitation mode :

#     11 = children living with two parents
#     12 = children living with one parent
#     21 = adults living in couple without child
#     22 = adults living in couple with children
#     23 = adults living alone with children
#     31 = persons not from family living in the home
#     32 = persons living alone

df_population['MOCO'] = df_population['MOCO'].astype('category')
df_population['MOCO'].cat.categories = [
"children living with two parents",
"children living with one parent",
"adults living in couple without child",
"adults living in couple with children",
"adults living alone with children",
"persons not from family living in the home",
"persons living alone"
]



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


# geoJSON files

# In[7]:


#https://gis.stackexchange.com/questions/90553/fiona-get-each-feature-extent-bounds
#Get the bounding boxes for each departement

def explode(coords):
    """Explode a GeoJSON geometry's coordinates object and yield coordinate tuples.
    As long as the input is conforming, the type of the geometry doesn't matter."""
    for e in coords:
        if isinstance(e, (numbers.Number)):
            yield coords
            break
        else:
            for f in explode(e):
                yield f

def bbox(f):
    x, y = zip(*list(explode(f['geometry']['coordinates'])))
    return min(x), min(y), max(x), max(y)

#Create DF with bounding boxes for each departement
df_bbox_departements = pd.DataFrame(columns = ['nom',
                                              'code',
                                              'bbox'])

for i in json_departements['features']:
    df_bbox_departements = df_bbox_departements.append({'nom':i['properties']['nom'],
                                 'code':i['properties']['code'],
                                 'bbox':bbox(i)}, ignore_index = True)
    
df_bbox_departements["min_x"] = df_bbox_departements.bbox.apply(lambda x: x[0])
df_bbox_departements["min_y"] = df_bbox_departements.bbox.apply(lambda x: x[1])
df_bbox_departements["max_x"] = df_bbox_departements.bbox.apply(lambda x: x[2])
df_bbox_departements["max_y"] = df_bbox_departements.bbox.apply(lambda x: x[3])


# In[8]:


# Create one bounding box for France
df_bbox_departements['dummy'] = 1

df_bbox_france = df_bbox_departements.groupby('dummy').agg({'min_x':{'min_x':'min'},
                         'min_y':{'min_y':'min'},
                         'max_x':{'max_x':'max'},
                         'max_y':{'max_y':'max'}})

df_bbox_france.reset_index()


# In[9]:


#Plot the departements
#These are the French equivalent of our TAs

map_departements = folium.Map(location=[48.864716, 2.349014])
folium.GeoJson(data = json_departements,
               name='features'
              ).add_to(map_departements)

list_departements = list(json_departements.values())

map_departements.fit_bounds(bounds = [
    [
     float(df_bbox_france['max_y']['max_y']),
     float(df_bbox_france['max_x']['max_x'])
    ],
    [
     float(df_bbox_france['min_y']['min_y']),
    float(df_bbox_france['min_x']['min_x'])
    ]])

map_departements


# In[10]:


#Test style functions
def style_function(feature):
    return {
        'fillColor': '#ffaf00',
        'color': 'blue',
        'weight': 1.5,
        'dashArray': '5, 5'
    }


def highlight_function(feature):
    return {
        'fillColor': '#ffaf00',
        'color': 'green',
        'weight': 3,
        'dashArray': '5, 5'
    }

#https://github.com/python-visualization/folium/issues/497
#Tooltips not supported from folium.GeoJson currently...
map_departements = folium.Map(location=[48.864716, 2.349014])
folium.GeoJson(data = json_departements,
               name='features',
               style_function = style_function,
               highlight_function = highlight_function
              ).add_to(map_departements)

map_departements


# In[ ]:


#Careful, this completely uses my ram/cpu
#There are > 30,000 shapes

# map_communes = folium.Map()
# folium.GeoJson(data = json_communes,
#                name='features'
#               ).add_to(map_communes)

# map_communes


# Analysis

# Distribution of firm sizes per departement

# In[10]:


#Create data for % in each firm size for each departement
df_dep_percentage = df_tranche.set_index(['CODGEO', 'LIBGEO', 'REG', 'DEP']).stack().reset_index()
df_dep_percentage = df_dep_percentage.rename(index = str,
                             columns = {'level_4':'firm_size_code',
                                        0:'value'})

df_dep_percentage = df_dep_percentage[df_dep_percentage['firm_size_code'] != 'E14TST']

df_dep_percentage = df_dep_percentage.groupby(["DEP", "firm_size_code"]).sum().loc[:,"value"].reset_index().merge(df_dep_percentage.groupby(["DEP"]).sum().loc[:,"value"].reset_index(),
      how = 'inner',
      on = "DEP")

df_dep_percentage = df_dep_percentage.rename(index = str,
                            columns = {"value_x":'value',
                            "value_y":'total'})

df_dep_percentage["percentage"] = df_dep_percentage['value'] / df_dep_percentage['total']
df_dep_percentage.head(10)

df_dep_percentage.firm_size_code = df_dep_percentage.firm_size_code.astype("category")
df_dep_percentage.firm_size_code = df_dep_percentage.firm_size_code.cat.reorder_categories(['E14TS0ND', 'E14TS1', 'E14TS6', 'E14TS10','E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500'])
df_dep_percentage.firm_size_code.cat.as_ordered(inplace = True)


# In[11]:


#To do: rather than have the numeric colour index map each to equally spaced values
#To do: why isn't this using the category order
df_dep_colour_map = df_dep_percentage.DEP.drop_duplicates()
df_dep_colour_map = df_dep_colour_map.reset_index()

fig, ax = plt.subplots()
ax.scatter(x =df_dep_percentage.firm_size_code,
          y = df_dep_percentage.percentage,
          s = 10000 * df_dep_percentage.value / np.sum(df_dep_percentage.value),
          c = df_dep_percentage.merge(df_dep_colour_map)['index'])

x_labels = ax.get_xticklabels()
plt.setp(x_labels, rotation=90)

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

ax.set(xlabel='Firm Size', 
       ylabel='% of firms in departement',
       title='Firm size distribution for each departement \n Departements are distinguished by colour')

plt.show()


# In[61]:


#The main difference is % of firms that are small.
#Let's cluster on this see if that's the case.

df_cluster = df_dep_percentage

#Drop the departements which aren't included in the GeoJson
list_geo_deps = []

for i in json_departements['features']:
    list_geo_deps.append(i['properties']['code'])

df_cluster = df_cluster[np.in1d(df_cluster.DEP,
                                list_geo_deps)]

df_result_dep = df_cluster.DEP.drop_duplicates()

# df_cluster.firm_size_code = df_cluster.firm_size_code.astype('str')
df_cluster = df_cluster.drop('value', axis = 1).set_index(['DEP', 'total', 'firm_size_code']).unstack()

n_clusters = 4

km_model = KMeans(n_clusters)

km_model_fit = km_model.fit(df_cluster)
km_model_prediction = km_model_fit.predict(df_cluster)

df_results = pd.DataFrame({'DEP':df_result_dep,
                          'cluster':km_model_prediction})
df_results


# In[62]:


#need to join the clustering to the departement geoJson

for i in range(len(json_departements['features'])):
    dep_to_use = json_departements['features'][i]['properties']['code']
    
    json_departements['features'][i]['properties'].    update({'cluster':(int(df_results[df_results.DEP == dep_to_use]['cluster']))})


# In[63]:


cmap = plt.get_cmap(lut = n_clusters)

def style_function(feature):
    
    fill_col = rgb2hex(cmap(feature['properties']['cluster']))
            
    return {
        'fillColor': fill_col,
        'fillOpacity': .9
    }
    

map_departements = folium.Map(location=[48.864716, 2.349014])
folium.GeoJson(data = json_departements,
               name='features',
               style_function = style_function
              ).add_to(map_departements)

map_departements


# In[64]:


#Means per cluster, and the departements included in each
#Mayotte doesn't have a polygon in the shapefile

for i in df_results.cluster.unique():
    print(
        (i, 
         df_geo[np.in1d(df_geo.numéro_département,
               df_results[df_results.cluster == i]['DEP'])]['nom_département'].unique(),
         df_cluster[np.in1d(df_cluster.index.get_level_values(0),
       df_results[df_results.cluster == i]['DEP'])].mean()
        )
    )


# How about population overlayed on the map?

# In[65]:


#Create the population per departement

#Careful there are leading 0s in df_tranche
df_tranche.CODGEO.unique()
df_population.CODGEO.unique()

df_tranche_popn = df_tranche.loc[:,['CODGEO', 'DEP']]
df_tranche_popn["CODGEO"] = [i.strip("0") for i in df_tranche_popn["CODGEO"]]

df_tranche_popn["CODGEO"].unique()

df_popn_codgeo = df_population.groupby("CODGEO").      agg({"NB":"sum"}).      reset_index()
df_popn_codgeo['CODGEO'] = df_popn_codgeo['CODGEO'].astype('str')

df_tranche_popn = df_tranche_popn.merge(df_popn_codgeo, on = "CODGEO").groupby("DEP").agg({"NB":"sum"})

df_tranche_popn = df_tranche_popn.reset_index()


# In[66]:


#Actually rather than joining to the JSON features we can use choropleth

map_departements = folium.Map(location=[48.864716, 2.349014])

map_departements.choropleth(
    data = df_tranche_popn, 
                      geo_data = json_departements, 
                      columns = ["DEP", "NB"],
                      key_on = 'feature.properties.code',
                      fill_color='YlGn',
                      fill_opacity=0.9
                           )
map_departements


# In[106]:


#What about salary

#SNHM14 is mean net salary 

df_salaire_dep = df_popn_codgeo.merge(df_salaire.loc[:, ["CODGEO", "SNHM14"]], on = "CODGEO").merge(df_tranche.loc[:, ["CODGEO", "DEP"]], on = "CODGEO")

df_salaire_dep = df_salaire_dep.merge(df_salaire_dep.groupby("DEP").agg({"NB":"sum"}).reset_index().rename({"NB":"NB_DEP"}, axis = 1), on = "DEP")

df_salaire_dep["DEP_POPN_PROP"] = df_salaire_dep["NB"] / df_salaire_dep["NB_DEP"]

df_salaire_dep["SALAIRE_AVG"] = df_salaire_dep["SNHM14"] * df_salaire_dep["DEP_POPN_PROP"]

df_salaire_dep = df_salaire_dep.groupby("DEP").agg({"SALAIRE_AVG":"sum"}).reset_index()

df_salaire_dep


# In[109]:


#Show me the money

map_departements = folium.Map(location=[48.864716, 2.349014])

map_departements.choropleth(
    data = df_salaire_dep, 
                      geo_data = json_departements, 
                      columns = ["DEP", "SALAIRE_AVG"],
                      key_on = 'feature.properties.code',
                      fill_color='YlGn',
                      fill_opacity=0.7
                           )
map_departements


# In[117]:


def salaire_viz_map(var_str):
    assert var_str in df_salaire.columns
#     assert  df_salaire.dtypes[var_str] is numeric
    
    df_salaire_dep = df_popn_codgeo.    merge(df_salaire.loc[:, ["CODGEO", var_str]], on = "CODGEO").    merge(df_tranche.loc[:, ["CODGEO", "DEP"]], on = "CODGEO")

    df_salaire_dep = df_salaire_dep.    merge(df_salaire_dep.groupby("DEP").agg({"NB":"sum"}).reset_index().rename({"NB":"NB_DEP"}, axis = 1), on = "DEP")

    df_salaire_dep["DEP_POPN_PROP"] = df_salaire_dep["NB"] / df_salaire_dep["NB_DEP"]

    df_salaire_dep["SALAIRE_AVG"] = df_salaire_dep[var_str] * df_salaire_dep["DEP_POPN_PROP"]

    df_salaire_dep = df_salaire_dep.groupby("DEP").agg({"SALAIRE_AVG":"sum"}).reset_index()
    
    map_departements = folium.Map(location=[48.864716, 2.349014])

    map_departements.choropleth(
        data = df_salaire_dep, 
        geo_data = json_departements, 
        columns = ["DEP", "SALAIRE_AVG"],
        key_on = 'feature.properties.code',
        fill_color='YlGn',
        fill_opacity=0.7)
    
    return(map_departements)


# In[119]:


#Best place to be an executive?
map_salaire_SNHMC14 = salaire_viz_map("SNHMC14")
map_salaire_SNHMC14


# In[120]:


#Best place to be a dude?
map_salaire_SNHMH14 = salaire_viz_map("SNHMH14") 
map_salaire_SNHMH14


# In[121]:


#Best place to be young

map_salaire_SNHM1814 = salaire_viz_map("SNHM1814") 
map_salaire_SNHM1814


# In[123]:


#Best place to be old?
map_salaire_SNHM5014  = salaire_viz_map("SNHM5014") 
map_salaire_SNHM5014 


# Paris is always the answer.
