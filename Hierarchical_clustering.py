#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES & DATASET

# In this step we import all libraires that is required to proces our datset. Following this we will import our dataset. And the dataset that we use is the Happiness Index dataset

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


#Importing Dataset
dataset = pd.read_csv("Happiness-Data.csv")
dataset


# # MISSING VALUES 

# In this stage we will check if there is any missing values that are present in our dataset. this step is essential as presence of mising values could be serious bottlenecks in the processing of our dataset

# In[3]:


#Finding Missing Values
print(dataset.isnull().sum())


# # DATATYPES

# The following step helps to identify the dtatype present in our dataset.

# In[4]:


#Knowing the datatypes
dataset.dtypes


# # CORRELATION

# In this step we analyse the correlation between features . this step is essential as it helps to identify the redundant features in our dataset

# In[5]:


#Finding the correlation between features using Heatmaps
plt.figure(figsize=(14,12))
sns.heatmap(dataset.corr(), annot = True)


# # DROPPING REDUNDANT FEATURES

# In this step,we filter out all the redundant features that are not necessary for our analysis. In the below step we take all those features that have correlation above 95%.

# In[6]:


#Finding the features to be dropped
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr


# In[7]:


#Numeric feature
num_features = dataset.select_dtypes(include=[np.number])
x= num_features
x


# In[8]:


#List of Features to be dropped
corr_features = correlation(x, 0.95)
len(set(corr_features))


# In[9]:


corr_features


# In[10]:


df_unclean = dataset.drop(labels=['Explained_by:Freedom_to_make_life_choices','Explained_by:Generosity','Explained_by:Healthy_life_expectancy', 'Explained_by:Log_GDP_per_capita','Explained_by:Perceptions_of_corruption','Explained_by:Social_support','Lowerwhisker','Upperwhisker', 'Ladder_score_in_Dystopia'],axis=1)
df_unclean


# In[11]:


df_unclean.describe()


# In[12]:


df_unclean.shape


# In[13]:


#Removing Outliers
df_unclean.dtypes


# # DROPPING STRINGS

# Since strings cannot be normalized, we will rmove all the strings in our dataset, before normalizing the data

# In[14]:


df = df_unclean.drop(labels=['Country_name','Regional_indicator'],axis=1)
df.shape


# # NORMALIZATION OF FEATURES

# Data Normalization is the process of rescaling the data in a dataset and improving its integrity by eliminating data redundancy.

# In[15]:


#Normalize the data attributes 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
df_scaled = pd.DataFrame(d, columns=names)
df_scaled.head()


# In[16]:


df_scaled


# # DIMENSIONALITY REDUCTION

#  Dimensionality Reduction is the process by which we convert a high dimensionality dataset into a low dimensionality dataset. In this step we aim to reduce our dataset into a lower dimensional dataset that would capture a variance above 90%.

# In[17]:


pca = PCA()
principalcomponents = pca.fit_transform(df_scaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained variance')
plt.show


# In[18]:


# Create a PCA instance: pca
pca = PCA(n_components= 4)
principalcomponents = pca.fit_transform(df_scaled)


# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[19]:


pca.explained_variance_


# In[20]:


df_scaled.shape


# In[21]:


principalcomponents.shape


# In[22]:


final_data = pd.DataFrame(principalcomponents)
final_data


# # TYPES OF DENDROGRAMS

# There are mainly 4 types of linkages used in Agglomerative clustering, namely – Complete, 
# Ward, Single and Average. In the figure below, we cans see the dendrograms with these
# linkages. Among these linkages ward linkage is giving a perfect division of clusters according
# to the dendrograms plotted. Moreover, Ward linkage can minimize  cluster
# variance and it is ideal for datasets with considerable number of outliers.

# In[23]:


#Plotting Dendrogram using Complete linkage
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms - Complete Linkage")
dend = shc.dendrogram(shc.linkage(final_data, method='complete'))


# In[24]:


#Plotting Dendrogram using Single linkage
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms - Single Linkage")
dend = shc.dendrogram(shc.linkage(final_data, method='single'))


# In[25]:


#Plotting Dendrogram using Average linkage
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms- Average Linkage")
dend = shc.dendrogram(shc.linkage(final_data, method='average'))


# In[26]:


#Plotting Dendrogram using Ward linkage

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms -Ward Linkage")
dend = shc.dendrogram(shc.linkage(final_data, method='ward'))


# # ANALYZING DENDROGRAMS

# In[27]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(final_data)


# In[28]:


df_clustered=df_unclean
df_clustered['clusters']=cluster.fit_predict(final_data).tolist()
df_clustered


# In[29]:


#Dystopia countries
df_clustered[df_clustered.clusters==1]


# In[30]:


df_clustered[df_clustered.clusters==1].shape


# In[31]:


#Utopian Countries
df_clustered[df_clustered.clusters==0]


# In[32]:


df_clustered.sort_values( by="Ladder_score",ascending=False)


# In[33]:


df_clustered.sort_values( by="Standard_error_of_ladder_score",ascending=False)


# In[34]:


df_clustered.sort_values( by="Logged_GDP_per_capita",ascending=False)


# In[35]:


df_clustered.sort_values( by="Generosity",ascending=False)


# In[36]:


df_clustered.sort_values( by="Social_support",ascending=False)


# In[37]:


df_clustered.sort_values( by="Healthy_life_expectancy",ascending=False)


# In[38]:


df_clustered.sort_values( by="Freedom_to_make_life_choices",ascending=False)


# In[39]:


df_clustered.sort_values( by="Perceptions_of_corruption",ascending=False)


# In[40]:


df_clustered.sort_values( by="Dystopia_plus_residual",ascending=False)


# In[ ]:




