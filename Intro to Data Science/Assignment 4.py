# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:10:38 2017

@author: Fionn Delahunty
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 300
random_state = 170

X, y = make_blobs(n_samples=n_samples, cluster_std=3.0, shuffle=False, random_state=random_state)
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()
kmeans = KMeans(n_clusters=3, init='random', max_iter=30, n_init=1, algorithm='full').fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)

if y_pred[1] == y_pred[2]:
    print "1 and 2 are in the same cluster"
else:
    print "1 and 2 are in different clusters" 


#%%
X2, y2 = make_blobs(n_samples=n_samples, cluster_std=1.5, shuffle=False, random_state=random_state)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X2, transformation)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1])
plt.show()
X_varied, y_varied = make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state, shuffle=False)
plt.scatter(X_varied[:, 0], X_varied[:, 1])
plt.show()
X_filtered = np.vstack((X2[:100], X2[101:131], X2[201:215]))
plt.scatter(X_filtered[:, 0], X_filtered[:, 1])
plt.show()
