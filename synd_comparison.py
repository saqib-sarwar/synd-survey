#!/usr/bin/env python
# coding: utf-8

# **Define Models to be used for Synthetic Data Generation - CTGAN and TVAE**

# In[5]:


import pandas as pd
import numpy as np
import time

from sdv.tabular import TVAE
from sdv.tabular import CTGAN
from sdv.evaluation import evaluate

import warnings
warnings.filterwarnings('ignore')

# In[3]:

models_tvae = []
models_ctgan = []

for i in range(6):
    models_ctgan.append(CTGAN(batch_size=300, verbose=True, epochs=80))
    models_tvae.append(TVAE(batch_size=300, epochs=80))


# In[6]:


df1 = pd.read_csv("dataset/adult.csv")
df1.name = "adult"

df2 = pd.read_csv("dataset/breast_cancer.csv")
df2.name = "breast_cancer"

df3 = pd.read_csv("dataset/heart_cleveland_upload.csv")
df3.name = "heart"

df4 = pd.read_csv("dataset/Iris.csv")
df4.name = "iris"

df5 = pd.read_csv("dataset/creditcard.csv")
df5.name = "credit"


# In[7]:


dfs = [df1, df2, df3, df4, df5]

# To store generated synthetic data
synthetic_data_mapping = {}

metrics=['CSTest', 'KSTest', 'ContinuousKLDivergence', 'DiscreteKLDivergence']
saved_models = {}


# In[8]:


for df in dfs:
    print('\n' + '%'*40)
    print('\033[1m' + df.name + '\033[0m')
    print('%'*40 + '\n')
    df.info()
    display(df.head())
    synthetic_data_mapping['syn_' + df.name] = []


# In[8]:


dfs=[df1, df2, df3, df4, df5]


# In[13]:



def evaluate_model(model, df):
    
    print('Training in Progress - ' + model.__class__.__name__ + '_' + df.name + '\n')
    
    # Record training time
    start = time.time()
    model.fit(df)
    end = time.time()
    
    print( '\n' + model.__class__.__name__ + ' trained. \nTraining time: ' + str(end-start) + ' seconds \n')
    syn_data = model.sample(len(df))
    syn_data.name = df.name + '-' + model.__class__.__name__

    # Save Generated Synthetic Data for each model in a dictionary 
    synthetic_data_mapping['syn_' + df.name].append(syn_data)    
    
    # Record evaluation time
    start = time.time()
    ee = evaluate(syn_data, df, metrics=metrics , aggregate=False)
    end = time.time()
    print("Synthetic Data Evaluation - " +  model.__class__.__name__ + '_' + df.name + '\n')
    display(ee)
    print('\nEvaluation time: ' + str(end-start) + ' seconds \n')
    
    
    # Save the model
    saved_model_name = model.__class__.__name__ + '_' + df.name + '.pkl'
    model.save(saved_model_name)
    saved_models[saved_model_name] = model


# In[14]:



k = 0;
for df in dfs:
    evaluate_model(models_ctgan[k], df)
    evaluate_model(models_tvae[k], df)
    k += 1


# In[15]:


saved_models


# In[25]:


get_ipython().run_line_magic('store', 'synthetic_data_mapping')


# In[26]:


get_ipython().run_line_magic('store', 'df1')
get_ipython().run_line_magic('store', 'df2')
get_ipython().run_line_magic('store', 'df3')
get_ipython().run_line_magic('store', 'df4')
get_ipython().run_line_magic('store', 'df5')


# In[29]:


get_ipython().run_line_magic('store', 'df1.name')
get_ipython().run_line_magic('store', 'df2.name')
get_ipython().run_line_magic('store', 'df3.name')
get_ipython().run_line_magic('store', 'df4.name')
get_ipython().run_line_magic('store', 'df5.name')


# In[17]:


synthetic_data_mapping['syn_' + df1.name][0].head()


# In[18]:


synthetic_data_mapping['syn_' + df1.name][1].name


# **Visualisation**

# In[22]:


import seaborn as sns
from matplotlib import pyplot as plt

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
sns.kdeplot(data=df1,ax=ax0, x='age', hue='income')
sns.kdeplot(data=synthetic_data_mapping['syn_' + df1.name][0],ax=ax1, x='age', hue='income', label='ctgan', ls='--')
sns.kdeplot(data=synthetic_data_mapping['syn_' + df1.name][1],ax=ax2, x='age', hue='income', label='tvae', ls='-.')
plt.show()


# In[23]:


obj_data = df1.select_dtypes(include=['object']).copy()
obj_data.head()


# In[ ]:


synthetic_data_mapping[df1.name][1].groupby(['income']).size()


# In[ ]:


synthetic_data_mapping[df1.name][0].groupby(['income']).size()


# In[ ]:


df1.groupby('income').size()


# In[24]:


# VBGMM
from sklearn.mixture import GaussianMixture

vbgmm = GaussianMixture(n_components=3, random_state=42)
col = df1['education.num'].values.reshape(-1,1)
vbg = vbgmm.fit(col)
vbg.means_.shape
vbg.means_


# **VAE for Adult Census Data**

# In[ ]:



"""import statsmodels.api as sm

m = sm.OLS.from_formula("income_code~ age + fnlwgt+ education_num + capital_gain + capital_loss + hours_per_week", merged_df)"""

