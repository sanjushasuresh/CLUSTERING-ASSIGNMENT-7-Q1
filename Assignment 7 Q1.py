# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:27:31 2022

@author: LENOVO
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("crime_data.csv")
df.head()
df.info()
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df

df.boxplot("Murder",vert=False)
df.boxplot("Assault",vert=False)
df.boxplot("UrbanPop",vert=False)
# There is no outliers
df.boxplot("Rape",vert=False)
Q1=np.percentile(df["Rape"],25)
Q3=np.percentile(df["Rape"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df["Rape"]>UW
df[df["Rape"]>UW]

df.drop([1,27],axis=0,inplace=True)
df.shape

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
X = norm_func(df.iloc[:,0:])
X.describe()


### K-Means clustering ###
# To check how many clusters are requried
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)
print(inertia)

# Elbow method to see variance in inertia by clusters
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()
# From the graph we can see that the optimal number of clusters is 4

# Scree plot
import seaborn as sns
d1 = {"kvalue": range(1, 11),'inertiavalues':inertia}
d2 = pd.DataFrame(d1)
sns.barplot(x='kvalue',y="inertiavalues", data=d2) # kvalue=clusters
# Here the variance in inertia b/w 4th and 5th cluster is less so we can go with 4 clusters

KM=KMeans(n_clusters=4,n_init=10,max_iter=300)
Y=KM.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
df1=pd.concat([X,Y],axis=1)


### Hierarchical clustering ###
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram 
z = linkage(X, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Crime')
sch.dendrogram(z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,)  # font size for the x axis labels
plt.show()
# In the graph, if we draw a line through the point 1.0 then we get 4 clusters

from sklearn.cluster import AgglomerativeClustering
AC=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')
Y=AC.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
df2=pd.concat([X,Y],axis=1)



### DBSCAN Clustering ###
# Normalize heterogenous numerical data using standard scalar fit transform to dataset
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
df_norm=StandardScaler().fit_transform(df)
df_norm
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(df_norm)
# Noisy samples are given the label -1.
dbscan.labels_
# Adding clusters to dataset
df['clusters']=dbscan.labels_
df
df.groupby('clusters').agg(['mean']).reset_index()
# The output are outliers/noise data after removing this we can apply any other 
# clustering techniques for better output

# Plot Clusters
plt.figure(figsize=(10,7))  
plt.scatter(df['clusters'],df['UrbanPop'],c=dbscan.labels_) 
plt.title('DBSCAN Clustering plot')
plt.xlabel('Clusters')
plt.ylabel('UrbanPop')

plt.figure(figsize=(10,7))
plt.scatter(df['clusters'],df['Murder'],c=dbscan.labels_) 
plt.title('DBSCAN Clustering plot')
plt.xlabel('Clusters')
plt.ylabel('Murder')

plt.figure(figsize=(10,7))
plt.scatter(df['clusters'],df['Assault'],c=dbscan.labels_) 
plt.title('DBSCAN Clustering plot')
plt.xlabel('Clusters')
plt.ylabel('Assault')

plt.figure(figsize=(10,7))
plt.scatter(df['clusters'],df['Rape'],c=dbscan.labels_) 
plt.title('DBSCAN Clustering plot')
plt.xlabel('Clusters')
plt.ylabel('Rape')
# Here the estimated number of clusters are 3

# In Hierarchical clustering, dendrograms helps in clear visualization  
# K-means is considered as the simplest and quickest one which guarantees convergence 
# and DBSCAN can handle the noise very well.