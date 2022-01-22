#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[10]:



fb = pd.read_csv('/Users/andrejkleonskij/Data Science Git/dataset_facebook_cosmetics.csv', sep = ';')

# разделите данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[13]:


sns.distplot(y_train)
plt.show()


# In[15]:


sns.distplot(y_test)
plt.show()


# In[18]:


# корреляционная матрица
corr_m = fb.corr()
plt.figure(figsize = (15,15))
sns.heatmap(fb.corr(), square=True,annot = True)
plt.show()

