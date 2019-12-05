#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cols = sns.color_palette()
d_cols = {0:cols[0],1:cols[1],2:cols[2],3:cols[3],4:cols[4]}


# ## Hyperparameters

# #### Example 1A: gaussian distributed

# In[21]:


X, y = make_blobs(n_samples=400, 
                  centers=np.array([[2,2],[7,7]]), 
                  cluster_std=[0.5,0.5], 
                  n_features=2,
                  random_state=0)

plt.close('all')
# plt.title('original', fontsize=40)
sns.scatterplot(X[:,0],X[:,1], hue=y, legend=False, palette=d_cols)
plt.xlim([0.0,9.0])
plt.ylim([0.0,9.0])
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('./plots_perplexity_gaussian/original.png')
plt.show()


# In[22]:


for p in [2,5,15,30,50,75,100]:
    
    tsne = TSNE(perplexity=p,
                learning_rate=200.0,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.close('all')
#     plt.title('perplexity: %d' %p, fontsize=40)
    sns.scatterplot(X_tsne[:,0],X_tsne[:,1], hue=y, legend=False, palette=d_cols)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('tSNE 1', fontsize=20)
    plt.ylabel('tSNE 2', fontsize=20)
    
    plt.savefig('./plots_perplexity_gaussian/perplexity%d.png' %p)
    plt.show()


# #### Example 1B: gaussian distributed - many samples

# In[5]:


X, y = make_blobs(n_samples=7000, 
                  centers=np.array([[2,2],[7,7]]), 
                  cluster_std=[0.5,0.5], 
                  n_features=2,
                  random_state=0)

plt.close('all')
# plt.title('original', fontsize=40)
sns.scatterplot(X[:,0],X[:,1], hue=y, legend=False, palette=d_cols)
plt.xlim([0.0,9.0])
plt.ylim([0.0,9.0])
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('./plots_perplexity_gaussian_7000/original.png')
plt.show()


# In[6]:


for p in [2,5,15,30,50,75,100]:
    
    tsne = TSNE(perplexity=p,
                learning_rate=200.0,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.close('all')
#     plt.title('perplexity: %d' %p, fontsize=40)
    sns.scatterplot(X_tsne[:,0],X_tsne[:,1], hue=y, legend=False, palette=d_cols)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('tSNE 1', fontsize=20)
    plt.ylabel('tSNE 2', fontsize=20)
    
    plt.savefig('./plots_perplexity_gaussian_7000/perplexity%d.png' %p)
    plt.show()


# #### Example 1C: gaussian distributed - few samples

# In[7]:


X, y = make_blobs(n_samples=30, 
                  centers=np.array([[2,2],[7,7]]), 
                  cluster_std=[0.5,0.5], 
                  n_features=2,
                  random_state=0)

plt.close('all')
# plt.title('original', fontsize=40)
sns.scatterplot(X[:,0],X[:,1], hue=y, legend=False, palette=d_cols)
plt.xlim([0.0,9.0])
plt.ylim([0.0,9.0])
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('./plots_perplexity_gaussian_30/original.png')
plt.show()


# In[8]:


for p in [2,5,15,30,50,75,100]:
    
    tsne = TSNE(perplexity=p,
                learning_rate=200.0,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.close('all')
#     plt.title('perplexity: %d' %p, fontsize=40)
    sns.scatterplot(X_tsne[:,0],X_tsne[:,1], hue=y, legend=False, palette=d_cols)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('tSNE 1', fontsize=20)
    plt.ylabel('tSNE 2', fontsize=20)
    
    plt.savefig('./plots_perplexity_gaussian_30/perplexity%d.png' %p)
    plt.show()


# ## Behaviour

# #### Example 1: uniformly distributed noise

# In[11]:


n_samples = 400
X = np.ndarray((n_samples, 2))
X[:,0] = np.random.uniform(0,5,n_samples)
X[:,1] = np.random.uniform(0,5,n_samples)
y = np.zeros(n_samples)
y[X[:,0]<5] = 4
y[X[:,0]<4] = 3
y[X[:,0]<3] = 2
y[X[:,0]<2] = 1
y[X[:,0]<1] = 0

plt.close('all')
# plt.title('original', fontsize=40)
sns.scatterplot(X[:,0],X[:,1], hue=y, legend=False, palette=d_cols)
plt.xlim([-1.0,6.0])
plt.ylim([-1.0,6.0])
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('./plots_perplexity_uniform/original.png')
plt.show()


# In[12]:


for p in [2,5,15,30,50,75,100]:
    
    tsne = TSNE(perplexity=p,
                learning_rate=200.0,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.close('all')
#     plt.title('perplexity: %d' %p, fontsize=40)
    sns.scatterplot(X_tsne[:,0],X_tsne[:,1], hue=y, legend=False, palette=d_cols)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('tSNE 1', fontsize=20)
    plt.ylabel('tSNE 2', fontsize=20)
    
    plt.savefig('./plots_perplexity_uniform/perplexity%d.png' %p)
    plt.show()


# #### Example 2: distance within clusters

# In[15]:


X, y = make_blobs(n_samples=400, 
                  centers=np.array([[3,3],[8,3]]), 
                  cluster_std=[0.8,0.1], 
                  n_features=2,
                  random_state=0)

print(np.unique(y))

plt.close('all')
sns.scatterplot(X[:,0],X[:,1], hue=y, legend=False, palette=d_cols)
plt.xlim([0.0,10.0])
plt.ylim([0.0,6.0])
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('./plots_cluster_size/original.png')
plt.show()


# In[16]:


for p in [2,5,15,30,50,75,100,150,200]:
    
    tsne = TSNE(perplexity=p,
                learning_rate=200.0,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.close('all')
#     plt.title('perplexity: %d' %p)
    sns.scatterplot(X_tsne[:,0],X_tsne[:,1], hue=y, legend=False, palette=d_cols)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('tSNE 1', fontsize=20)
    plt.ylabel('tSNE 2', fontsize=20)
    
    plt.savefig('./plots_cluster_size/perplexity%d.png' %p)
    plt.show()


# #### Example 3: distance between clusters

# In[19]:


X, y = make_blobs(n_samples=200, 
                  centers=np.array([[2,2],[2,4],[8,3]]), 
                  cluster_std=[0.2,0.2,0.2], 
                  n_features=2,
                  random_state=0)

print(np.shape(y))

plt.close('all')
sns.scatterplot(X[:,0],X[:,1], hue=y, legend=False, palette=d_cols)
plt.xlim([0.0,10.0])
plt.ylim([0.0,6.0])
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('./plots_cluster_distances/original.png')
plt.show()


# In[20]:


for p in [2,5,15,30,50,75,100,150,200]:
    
    tsne = TSNE(perplexity=p,
                learning_rate=200.0,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.close('all')
#     plt.title('perplexity: %d' %p)
    sns.scatterplot(X_tsne[:,0],X_tsne[:,1], hue=y, legend=False, palette=d_cols)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('tSNE 1', fontsize=20)
    plt.ylabel('tSNE 2', fontsize=20)
    
    plt.savefig('./plots_cluster_distances/perplexity%d.png' %p)
    plt.show()

