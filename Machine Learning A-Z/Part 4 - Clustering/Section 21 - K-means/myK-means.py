# K-means clustering

__author__ = 'James Schnebly'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# use elbow method to figure out number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
	kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, random_state = 0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply k-means to mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, random_state = 0)
Y_kmeans = kmeans.fit_predict(X)


# Vizualization for 2 dimesnsional clustering
plt.scatter(X[Y_kmeans == 0,0], X[Y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[Y_kmeans == 1,0], X[Y_kmeans == 1,1], s = 100, c = 'yellow', label = 'Cluster 2')
plt.scatter(X[Y_kmeans == 2,0], X[Y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[Y_kmeans == 3,0], X[Y_kmeans == 3,1], s = 100, c = 'blue', label = 'Cluster 4')
plt.scatter(X[Y_kmeans == 4,0], X[Y_kmeans == 4,1], s = 100, c = 'pink', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, c = 'magenta', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()