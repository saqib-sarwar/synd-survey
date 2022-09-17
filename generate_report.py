#!/usr/bin/env python
# coding: utf-8

# Generate Comparative report for all the synthetic data

# In[39]:


import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from cProfile import label
from matplotlib.pyplot import title




# In[62]:


class Report:

    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def get_plot_params(self,n_plots):
        n_cols = 3
        n_rows = int(n_plots/n_cols) if (n_plots%n_cols == 0) else int(n_plots/n_cols + 1)
        return n_rows, n_cols

    def generate_plots_numerical(self):
        num_cols = [col for col in self.df1.columns if self.df1.dtypes[col] in ['float64', 'int64']]
        
        n_rows, n_cols = self.get_plot_params(len(num_cols))
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,7*n_rows))
        i,j = 0, 0
        for col in num_cols:
            sns.kdeplot(data=self.df1, ax=ax[i][j], x=col, label="Real")
            sns.kdeplot(data=self.df2, ax=ax[i][j], x=col, label="CTGAN")
            ax[i][j].legend()
            j = (j+1)%3
            if(j==0): i += 1

    def generate_plots_categorical(self):
        cat_cols = [col for col in self.df1.columns if self.df1.dtypes[col] in ['object', 'category']]

        if len(cat_cols) > 1:

            n_rows, n_cols = self.get_plot_params(len(cat_cols))

            fig, ax = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,7*n_rows))
            i,j = 0, 0
            for col in cat_cols:

                # Convert object to categorical type
                self.df1[col] = self.df1[col].astype("category")
                # Plot histogram of real data for each column
                sns.histplot(data=self.df1, ax=ax[i][j], x=col, label="Real", color="grey")

                # Plot histogram of Synthetic data
                sns.histplot(data=self.df2, ax=ax[i][j], x=col, label="CTGAN", color="skyblue")
                # Label histogram for Synthetic Model used
                ax[i][j].legend()
                # Rotate ticks
                ax[i][j].tick_params(labelrotation=90)
                
                j = (j+1)%3
                if(j==0): i += 1
        else:
            pass

    def generate_corr_heatmap(self):
        
        # Correlation Matrix for each feature
        fig, ax = plt.subplots(1, 2, figsize=(15,7))
        corrmat = round(self.df1.corr(), 2)
        top_corr_features = corrmat.index
        #plot heat map
        sns.heatmap(self.df1[top_corr_features].corr(), ax=ax[0], annot=False,fmt='.2f',cmap="BuGn")

        corrmat = round(self.df2.corr(), 2)
        top_corr_features = corrmat.index
        #plot heat map
        sns.heatmap(self.df2[top_corr_features].corr(), ax=ax[1], annot=False,fmt='.2f',cmap="BuGn")
        return fig
    