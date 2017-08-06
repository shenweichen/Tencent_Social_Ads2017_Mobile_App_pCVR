
# coding: utf-8

# In[11]:

import os,gc
import pandas as pd
import numpy as np
import scipy.stats as sps
from tqdm import tqdm
from feature_joint import addTime
from utils import raw_data_path,feature_data_path


# In[4]:

def generate_click_trick():
    #df['origin_index'] = df.index
    feature_path = feature_data_path + 'global_tricks.pkl'
    if os.path.exists(feature_path):
        print('found '+feature_path)
    else:
        train = pd.read_pickle(raw_data_path+'train.pkl')
        test = pd.read_pickle(raw_data_path+'test.pkl')
        df = train.append(test)
        df = df[['global_index','creativeID','userID','label','clickTime',]]
        del train,test
        df = addTime(df)
        gc.collect()
        uct_cnt = df.groupby(['userID', 'creativeID']).size().reset_index()
        uct_cnt.rename(columns={0: 'global_uct_cnt'}, inplace=True)
        df = pd.merge(df, uct_cnt, how='left', on=['userID', 'creativeID'])

        df_1 = df.sort_values(by=['userID', 'clickTime'], ascending=True)
        first = df_1.drop_duplicates('userID')
        first['global_first'] = 1
        first = first[['userID','clickTime','global_first']]
        df = pd.merge(df,first,how='left',on=['userID','clickTime'])

        df_2 = df.sort_values(by=['userID', 'clickTime'], ascending=False)
        last = df_2.drop_duplicates('userID')
        last['global_last'] = 1
        last = last[['userID', 'clickTime', 'global_last']]
        df = pd.merge(df, last, how='left', on=['userID', 'clickTime'])
        pd.to_pickle(df[['clickDay','global_uct_cnt','global_first','global_last',]],feature_path)


# In[21]:

def add_click_trick(data,start_day,end_day):
    feature_path = feature_data_path + 'global_tricks.pkl'
    feature_names = ['global_uct_cnt','global_first','global_last']
    trick_final = pd.read_pickle(feature_path)
    trick_final = trick_final.loc[(trick_final.clickDay>=start_day)&(trick_final.clickDay<=end_day),feature_names]
    trick_final.index = data.index
    data = pd.concat([data,trick_final[feature_names]],axis=1)
    #data = pd.merge(data,trick_final,'left','global_index')
    return data


# In[4]:

if __name__ == '__main__':
    generate_click_trick()

