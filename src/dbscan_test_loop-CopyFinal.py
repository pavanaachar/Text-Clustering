
# coding: utf-8

# In[171]:


from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import operator
import dbscan2
from helper import Helper
from data_preprocessing import DataPreprocessing
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from scipy.spatial.distance import cdist
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as silhouette_score
from scipy.sparse import csr_matrix

import pandas as pd # pul
import matplotlib.pyplot as plt


# In[134]:


helper = Helper('train.dat')


# In[135]:


train = helper.get_data()


# In[136]:


indices, values = helper.separate_word_frequency(train)


# In[137]:


preprocessor = DataPreprocessing()


# In[138]:


mat = preprocessor.csr_build(train, indices, values)


# In[139]:


preprocessor.csr_info(mat)


# In[140]:


mat_idf = preprocessor.csr_idf(mat, copy=True)


# In[141]:


mat_norm = preprocessor.csr_l2normalize(mat_idf, copy=True)
# mat_norm = preprocessor.csr_l2normalize(mat, copy=True)


# In[142]:


preprocessor.csr_info(mat_norm)
# preprocessor.csr_info(mat_idf)


# In[143]:


# csr_arr = mat_norm.toarray()


# In[144]:


svd = TruncatedSVD(n_components=3)


# In[145]:


# svd.fit(mat_norm)
dr_mat = svd.fit_transform(mat_norm)


# In[146]:


print svd.explained_variance_ratio_.cumsum()
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[147]:


# dr_mat = svd.transform(mat_norm)
# dr_mat = svd.transform(mat)


# In[148]:


dr_mat
type(dr_mat)


# In[149]:


# centers = [[5, 5], [-5, -5], [-5, 5]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.9,
#                             random_state=0)

# X = StandardScaler().fit_transform(X)


# In[150]:


X = dr_mat


# In[172]:


sim_dist = cosine_distances(X, X)                                       #, 'cosine')


# In[152]:



import pandas as pd # pul
import matplotlib.pyplot as plt


# In[153]:


def determine_eps(sim_mat, minpts):
    eps_a = []
    for minpt in minpts:
        eps_arr = []
        for i in range(len(sim_mat)):
            eps_arr.append(np.sort(sim_mat[i])[minpt])
        eps_a.append(np.sort(eps_arr))

    eps = []
    for i in range(len(eps_a)):
        values= eps_a[i]

        #get coordinates of all the points
        nPoints = len(values)
        allCoord = np.vstack((range(nPoints), values)).T
        #np.array([range(nPoints), values])

        # get the first point
        firstPoint = allCoord[0]
        # get vector between first and last point - this is the line
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))

        # find the distance from each point to the line:
        # vector between all points and first point
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(vecFromFirst *
        np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
        idxOfBestPoint = np.argmax(distToLine)
        print "Min Pts value = ", minpts[i]
        print "Knee of the curve is at index =",idxOfBestPoint
        print "Knee value =", round(values[idxOfBestPoint],4)
        eps.append(round(values[idxOfBestPoint],4))

    # plot of the original curve and its corresponding distances
#     plt.figure(figsize=(12,6))
        plt.plot(values,label='Distance to Nearest Neighbors',color='b')
        plt.plot([idxOfBestPoint], values[idxOfBestPoint], marker='o',markersize=8, color="red", label='Knee')
        plt.xlabel("Points sorted according to the distance of "+str(minpts[i])+" nearest neighbor")
        plt.ylabel(str(minpts[i])+" nearest neighbor distance")
        plt.legend()
        plt.show()
    return eps


# In[173]:


# X = [tuple(x) for x in dr_mat[:500]]
# X = dr_mat[:4000]

# eps = [0.0307]
# eps = [0.008, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.018, 0.02]
# minpts = [21]

# minpts = [3,5,7]
minpts = [3,5,7,9,11,13,15,17,19,21]
# minpts = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
eps = determine_eps(sim_dist,minpts)  



# In[174]:


eps


# In[185]:


eps = [0.0008]
minpts = [25]


# In[186]:


label_list = {}

for e,m in zip(eps, minpts):
    dbs = dbscan2.DBSCAN(X, e, m, sim_dist)
    dbs.classify_points()
    dbs.connect_core_points()
    dbs.connect_border_points()
    dbs.assign_labels()
    labels = dbs.get_labels()
    # label_list.append(labels)
    label_list[e] = labels


# In[191]:


silhouettes = []

for e,m in zip(eps, minpts):
    print label_list[e]
    try:
        sil_score = silhouette_score(X, label_list[e], metric='cosine', sample_size=4000)
        silhouettes.append(sil_score)
    except ValueError:
        silhouettes.append(-1)

    with open('sub12/labels_new_' + str(e) + '_' + str(m) + '.dat', 'w+') as f:
        for i in labels:
            f.write("%d\n"%(i))


# In[188]:


silhouettes


# In[132]:


plt.plot(silhouettes)
plt.xlabel("silhouette score for different minpts,eps pairs")

plt.ylabel("silhouette coefficient")
plt.show()


# In[116]:


silhouettes.index(max(silhouettes))


# In[117]:


print silhouettes[4], minpts[4], eps[4]


# In[189]:


labels = label_list[0.0008]


# In[190]:


import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = [x for x in set(labels)]
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
print unique_labels, colors
k_c_map = {}
for k, col in zip(unique_labels,colors):
    k_c_map[k] = col

for i in range(len(X)):
    x,y,z = [x for x in X[i]]
    k = labels[i]
    col =  k_c_map[k]
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = X[]
    plt.plot(x, y, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % max(labels))
plt.show()

for i in range(len(X)):
    x,y,z = [x for x in X[i]]
    k = labels[i]
    col =  k_c_map[k]
    if k == max(labels):
        # Black used for noise.
        col = [0, 0, 0, 1]
    plt.plot(z, y, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % max(labels))
plt.show()


for i in range(len(X)):
    x,y,z = [x for x in X[i]]
    k = labels[i]
    col =  k_c_map[k]
    if k == max(labels):
        # Black used for noise.
        col = [0, 0, 0, 1]
    plt.plot(y, z, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % max(labels))
plt.show()


for i in range(len(X)):
    x,y,z = [x for x in X[i]]
    k = labels[i]
    col =  k_c_map[k]
    if k == max(labels):
        # Black used for noise.
        col = [0, 0, 0, 1]
    plt.plot(x, z, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % max(labels))
plt.show()


for i in range(len(X)):
    x,y,z = [x for x in X[i]]
    k = labels[i]
    col =  k_c_map[k]
    if k == max(labels):
        # Black used for noise.
        col = [0, 0, 0, 1]
    plt.plot(z, x, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % max(labels))
plt.show()


# In[109]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

unique_labels = [x for x in set(labels)]
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
print unique_labels, colors
k_c_map = {}
for k, col in zip(unique_labels,colors):
    k_c_map[k] = col

for i in range(len(X)):
    x,y,z = [x for x in X[i]]
    k = labels[i]
    col =  k_c_map[k]
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    ax.scatter(x, y, z, c=[0, 0, 1, 1])
plt.show()

