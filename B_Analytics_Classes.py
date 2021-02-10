
###############################################################
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from numpy import *
from datetime import datetime, date, timedelta, time
from operator import attrgetter

from sklearn.cluster import KMeans
###############################################################


## class to select the top n performing channels and add them to dataframe
## arguments: n:number of top channels to select, df_ch: dataframe of channels,
## df: dataframe to add top channels to
class AddTopChannels():
    def __init__(self,n,top_name,df_ch,df):
        self.n = n
        self.top_name = top_name
        self.df_ch = df_ch
        self.df = df
    def topn_channels(self):
        order = argsort(-self.df_ch.values, axis=1)[:, :self.n]
        arr = [self.df_ch.columns[k] for k in order]
        topn = pd.DataFrame(arr, columns=['{}_top{}'.format(self.top_name,i) for i in range(1, self.n+1)],index=self.df_ch.index)
        return topn
    def add_to_df(self):
        topn = self.topn_channels()
        df_out = pd.concat([self.df, topn], axis=1)
        return df_out 
        

## Class for cohort analysis
class AnalyseCohorts():
    def __init__(self,df,usr_id,date,period,aggr_col,aggr_type):
        self.df_c = df.copy()
        self.usr_id = usr_id
        self.date = date
        self.period = period
        self.aggr_col = aggr_col
        self.aggr_type = aggr_type
    def det_cohorts(self):
        self.df_c['T_period'] = self.df_c[self.date].dt.to_period(self.period)
        self.df_c['Cohort'] = self.df_c.groupby(self.usr_id)[self.date].transform('min').dt.to_period(self.period) 
        df_cohort = self.df_c.groupby(['Cohort', 'T_period']).agg(aggr_T=(self.aggr_col, self.aggr_type)).reset_index(drop=False)
        df_cohort['P_number'] = (df_cohort['T_period'] - df_cohort['Cohort']).apply(attrgetter('n'))
        cohort_pivot = df_cohort.pivot_table(index = 'Cohort',columns = 'P_number',values = 'aggr_T')
        return cohort_pivot
    def det_retention(self):
        cohort_pivot = self.det_cohorts()
        cohort_size = cohort_pivot.iloc[:,0]
        r_matrix = cohort_pivot.divide(cohort_size, axis = 0)
        return cohort_size, r_matrix
    def det_churn(self):
        cohort_size, r_matrix = self.det_retention()
        c_matrix = 1-r_matrix
        return cohort_size, c_matrix
