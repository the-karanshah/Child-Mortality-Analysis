"""
 > Brief Description: To identify the district which has the least child mortality, use it as the benchmark, identify the similarity of other districts with the benchmark, and then using the features which are least similar between the two districts as suggestions for improvements to reduce the mortality.

 > Topics from Courses:
 1. Spark
 2. Similarity Search using Cosine Similarity

 > Type of system used:
 1. Google Cloud DataProc (Image Version: 1.4.27-debian9)
	1 e2-standard-2 master node
	2 e2-highmem-4 worker nodes
 2. Google Colaboratory

"""

from pyspark.sql import SparkSession
from pyspark import SparkContext

sc = SparkContext.getOrCreate()


# libraries import 
import pandas as pd
import numpy as np
import sys
from scipy.stats import zscore
# full print
pd.options.mode.chained_assignment = None

# custom function to get average of features and mortality rate
def getCount(x):
  key = x[0]
  val = x[1]
  temp = list(val)
  lenrecord = len(temp[0])
  
  deaths =0
  for i in val:
    deaths = deaths + int(i[-1])
  avg_arr = np.array(list(val)).astype(np.float)
  centroid = np.mean(avg_arr, axis=0)
  population = avg_arr.shape[0]

  return (key,(population,deaths,deaths*1000/population,centroid,val))

 
# custom function to map from district to feature as the key
def remap(record):
  district = record[0] # district code
  features = record[1][3] # list of values for every col
  pop = record[1][0]
  deaths = record[1][1]
  mortality = record[1][2]
  return [(i, (x, (district,pop, deaths, mortality))) for i, x in enumerate(features)]


# custom function to perform feature normalization before similarity search
def normalize(record):
  featureNum = record[0]
  featureval = record[1]

  listOfVal =[]
  for i in featureval:
    listOfVal.append(i[0])

  listOfVal = np.array(listOfVal)
  normalized = zscore(listOfVal)

  toReturn = []
  for i in range(len(featureval)):
    tup = (featureNum, normalized[i], featureval[i][1])
    toReturn.append(tup)
  return toReturn


# custom function to perform cosine similarity operation between two district features
def cos_similarity(record):
  ideal = ideal_distr.value[0][1]
  ideal_arr = np.zeros(len(ideal))
  for tup in ideal:
    idx = tup[0]
    val = tup[1]
    ideal_arr[idx] = val
  district = record[0]
  dist = record[1]
  dist_arr = np.zeros(len(dist))
  for tup in dist:
    idx = tup[0]
    val = tup[1]
    dist_arr[idx] = val
  # cosine similarity computation
  idealRss = np.sqrt(np.sum(ideal_arr**2))
  distRss = np.sqrt(np.sum(dist_arr**2))
  xiyi = np.sum(ideal_arr*dist_arr)
  sim_metric = xiyi / (idealRss * distRss)

  return (district, sim_metric, dist)


# custom function to remap from feature number to district as the key
def remapToDistrict(record):
  featureNum = record[0]

  normalisedFeatureVal = record[1][0]

  rest = record[1]

  for i in rest:
    district = i[1][0]

# return [(i, (x, (district,pop, deaths, mortality))) for i, x in enumerate(features)]
  return [(x[0],(featureNum , normalisedFeatureVal, (x[1],x[2]))) for i,x in enumerate(rest)]


# custom function to perform remapping of keys and one of the values for grouping by
def remapper2(record):
  arr = []
  for tuple in record:
    idx = tuple[0]
    rest = tuple[1]
    for t in tuple[1][1]:
      district = t[0]
      pop = t[1]
      death = t[2]
      mort = t[3]
      temp = (district,(pop,death,mort, (idx,tuple[1][0])))
      arr.append(temp)
  return arr
    # val(district, (idx))
    # arr.append(val)

import sys

# Rdd transformation to perform averaging of all features in each district
# and then grouping them
rdd = sc.textFile(sys.argv[1]) \
        .map(lambda l : l.split(',')) \
        .filter(lambda x : x[0]!='state') \
        .map(lambda x : (x[1], x[2:4]+x[6:])) \
        .groupByKey() \
        .map(lambda x : getCount(x)) \
        .flatMap(lambda x: remap(x)) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1])))
        #.map(lambda x: normalize(x))\
        #.flatMap(lambda x : remapper2(x)) 
        # feature, [(district, normalized value)]
        # district, [(index, normalized features value)]
        # groupByDistrict [normalized features]

"""### Grouped by feature... for every district"""

rdd.take(4)

# normalized up to here -- performing normalization using the custom function
rdd_normalized = rdd.flatMap(lambda x:normalize(x))

rdd_normalized.take(4)


# another custom function to perform remapping for new key creation for grouping by
def reMapperNew(record):
  feat = record[0]
  normval = record[1]
  rest = record[2]
  toReturn = (rest[0], (feat, normval, rest[1:]))
  return toReturn

rddNorm_dist = rdd_normalized.map(lambda x: reMapperNew(x))
rddNorm_dist.take(5)

rdNorm_distgroup = rddNorm_dist.groupByKey() \
                                .map(lambda x: (x[0], list(x[1]))) \
                                .sortBy(lambda x: x[1][0][2][2])
rdNorm_distgroup.take(3)


# creating a broadcast variable to store the ideal district tuple
ideal_distr = sc.broadcast(rdNorm_distgroup.take(1))
print(ideal_distr.value[0][1])

# Rdd storing the cosine similarity between each district using their features
rdd_sim = rdNorm_distgroup.map(lambda x: cos_similarity(x))
rdd_sim.take(5)

"""### Let's see which districts are the farthest from the ideal district and what's the cosine similarity"""

print("\n\nLet's see which districts are the farthest from the ideal district and what's the cosine similarity")
print(rdd_sim.sortBy(lambda x:x[1]).take(5))
