# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:49:15 2020

@author: hossein
"""

#import geopandas

import descartes
from descartes import PolygonPatch
from sklearn import preprocessing
import matplotlib as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
#from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import confusion_matrix
import seaborn
from sklearn.ensemble import AdaBoostClassifier


year = "2017"



df = pd.read_csv("ConstituencyDataEngland.csv", sep=",", header=None)

#Reading in the Brexit vote data from CSV
brexit = pd.read_csv("Brexit.csv", sep=",", header=None)



df = pd.merge(df, brexit, on=0)

#Setting column names
df = df.rename(columns={"1_y": "Brexit", 0: "Constituencies", "1_x": "HousePrices", 2: "Wage",3: "British",4: "Tenure",5: "School Grades", 6:"0-9",7:"10-19",8:"20-29",9:"30-39",10:"40-49",11:"50-59",12:"60-69",13:"70-79",14:"80+",15:"Voted Leave"})
df = df[1:] #take the data less the header row

#Removing Constituency Column in prep for clustering
unscaled = df.drop(columns="Constituencies")

#Scaling the data so all attributes are weighted equally

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(unscaled.values)
scaled_values = pd.DataFrame(x_scaled)
mat = scaled_values.values









af = AffinityPropagation().fit(scaled_values)
cluster_centers_indices = af.cluster_centers_indices_
affinitylabels = af.labels_

#Number of clusters determined by the Preference parameter.
#This parameter sets how strongly each datapoint thinks it is an examplar

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)

#Inserting the affinitypropagation data into the dataframe
df.insert(0, "affinity", affinitylabels)

#df.to_csv('affinity_cluster.csv')

results2017 = pd.read_csv("2017Results.csv", sep=",", header=None)
results2017 = results2017.rename(columns={0: "Constituencies", 1: "Party"})

le = preprocessing.LabelEncoder()

x=le.fit_transform(results2017.values)





