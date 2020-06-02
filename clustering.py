"""
 > Brief Description: To validate the results we obtained using similarity analysis, we also performed clustering on our dataset to see which secondary indicators may be impacting mortality. 
 
 > Topics from Courses:
 1. Similarity analysis using Clustering

 > Type of system used:
 1. Google Cloud DataProc (Image Version: 1.4.27-debian9)
	1 e2-standard-2 master node
	2 e2-highmem-4 worker nodes
 2. Google Colaboratory

"""


# libraries import 
import pandas as pd
import numpy as np
import sys
from kmodes.kprototypes import KPrototypes
pd.options.mode.chained_assignment = None


# import data
df = pd.read_csv(sys.argv[1])

"""# analysis"""
df.columns
df.head(5)

"""## Clustering"""

# Taking a subset of significant and non-significant features because usually a large number of features give random clusters
subcolumn = ['highest_qualification', 'smoke', 'alcohol', 'chew', 'is_water_filter', 'lighting_source', 'counselled_for_menstrual_hyg',
             'iscoveredbyhealthscheme', 'urban', 'district', 'household_have_electricity', 'aware_abt_hiv', 'totalDeaths', 'age']

subset = df[subcolumn]
subset.head(5)

subset.shape

# Kprototype Clustering library for categorical and numerical features
dataset = subset.sample(n=50000, random_state=234)
kproto = KPrototypes(n_clusters=4, init='Cao', verbose=2)
#clusters = km.fit_predict(subset, categorical=list(range(18)))
#clusters = kproto.fit_predict(subsetKP, categorical=list(range(11)))
clusters = kproto.fit_predict(dataset, categorical=list(range(12)))

print(kproto.cluster_centroids_)

print("['highest_qualification', 'smoke', 'alcohol', 'chew', 'is_water_filter', 'lighting_source', 'counselled_for_menstrual_hyg', \
             'iscoveredbyhealthscheme', 'urban', 'modern_methods_used', 'household_have_electricity', 'aware_abt_hiv', 'totalDeaths', 'age']")
### Highest Qualification seems to be one distinguishing factor for lower mortality (Val = 0), another one is age (younger age corresponds to lower mortality)

print("Highest Qualification seems to be one distinguishing factor for lower mortality (Val = 0), another one is age (younger age corresponds to lower mortality)")