import pandas as pd
import numpy as np
import time

from sdv.tabular import TVAE
from sdv.tabular import CTGAN
from sdv.evaluation import evaluate

import warnings
warnings.filterwarnings('ignore')

class SDVModel:

    def __init__(self, model, df):
        self.model = model
        self.df = df
        self.syn_data = pd.DataFrame()

    def train_model(self):
        print('Training in Progress - ' + self.model.__class__.__name__ + '_' + self.df.name + '\n')
        # Record training time
        start = time.time()
        self.model.fit(self.df)
        end = time.time()
        print( '\n' + self.model.__class__.__name__ + ' trained. \nTraining time: ' + str(end-start) + ' seconds \n')

    def generate_sample(self):
        self.syn_data = self.model.sample(len(self.df))
        self.syn_data.name = self.df.name + '-' + self.model.__class__.__name__
        return self.syn_data

    def evaluate_model(self, metrics):
        # Record evaluation time
        start = time.time()
        ee = evaluate(self.syn_data, self.df, metrics=metrics , aggregate=False)
        end = time.time()
        print("Synthetic Data Evaluation - " +  self.model.__class__.__name__ + '_' + self.df.name + '\n')
        display(ee)
        print('\nEvaluation time: ' + str(end-start) + ' seconds \n')
        
    def save_model(self):
        # Save the model
        saved_model_name = self.model.__class__.__name__ + '_' + self.df.name + '.pkl'
        self.model.save(saved_model_name)
