
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import gc
import os
from smooth import BayesianSmoothing
from tqdm import tqdm
from utils import raw_data_path,feature_data_path,load_pickle,dump_pickle
from feature_joint import addAd,addPosition,addTime


# In[2]:

def trick(row):
    if row['ua_cnt'] <= 1:
        return 0
    elif row['ua_first'] > 0:
        return 1
    elif row['ua_last'] > 0:
        return 2
    else:
        return 3

def add_trick(df):
    ua_cnt = df.groupby(['userID', 'advertiserID']).size().reset_index()
    ua_cnt.rename(columns={0: 'ua_cnt'}, inplace=True)
    ua_cnt = ua_cnt[['userID', 'advertiserID', 'ua_cnt']]
    df = pd.merge(df, ua_cnt, how='left', on=['userID', 'advertiserID'])

    sorted = df.sort_values(by=['userID', 'advertiserID', 'clickTime'], ascending=True)
    first = sorted.drop_duplicates(['userID', 'advertiserID'])
    last = sorted.drop_duplicates(['userID', 'advertiserID'], keep='last')

    first['ua_first'] = 1
    first = first[['ua_first']]
    df = df.join(first)

    last['ua_last'] = 1
    last = last[['ua_last']]
    df = df.join(last)

    df['trick'] = df.apply(trick, axis=1)
    return df

def add_diff(df):
    sorted = df.sort_values(by=['userID', 'advertiserID', 'clickTime'], ascending=True)
    first = sorted.groupby(['userID', 'advertiserID'])['clickTime'].first().reset_index()
    first.rename(columns={'clickTime': 'first_diff'}, inplace=True)
    last = sorted.groupby(['userID', 'advertiserID'])['clickTime'].last().reset_index()
    last.rename(columns={'clickTime': 'last_diff'}, inplace=True)
    df = pd.merge(df, first, 'left', on=['userID', 'advertiserID'])
    df = pd.merge(df, last, 'left', on=['userID', 'advertiserID'])
    df['first_diff'] = df['clickTime'] - df['first_diff']
    df['last_diff'] = df['last_diff'] - df['clickTime']
    return df

def add_install2click(df ,i,actions):
    install2click = actions[actions.installTime < i*1000000]
    df = pd.merge(df, install2click, 'left', ['userID', 'appID'])
    df['install2click'] = df['clickTime'] - df['installTime']
    return df

def gen_tricks(start_day,end_day):
    """
    生成trick,first_diff,last_diff，install2click，根据gloabl_index拼接
    """
    train_data = load_pickle(raw_data_path+'train.pkl')
    test_data = load_pickle(raw_data_path+'test.pkl')
    actions = load_pickle(raw_data_path+'user_app_actions.pkl')
    data = train_data.append(test_data)
    del train_data,test_data
    data = addTime(data)
    data = addAd(data)
    
    for day in tqdm(np.arange(start_day, end_day+1)):
        feature_path = feature_data_path + 'tricks_day_'+str(day)+'.pkl'
        if os.path.exists(feature_path):
            print('found '+feature_path)
        else:
            print('generating '+feature_path)
            df = data.loc[data.clickDay == day]
            df = add_trick(df)
            df = add_diff(df)
            df = add_install2click(df, day,actions)
            dump_pickle(df[['global_index','trick','first_diff','last_diff','install2click']],feature_path)
            
def add_tricks(data):
    """
    
    """
    tricks = None
    for day in tqdm((data.clickTime//1000000).unique()):
        feature_path = feature_data_path + 'tricks_day_'+str(day)+'.pkl'
        if not os.path.exists(feature_path):
            gen_tricks(day,day)
        day_tricks = load_pickle(feature_path)
        if tricks is None:
            tricks = day_tricks
        else:
            tricks = pd.concat([tricks,day_tricks],axis=0)
    data = pd.merge(data,tricks,'left','global_index')
    return data


# In[6]:

if __name__ =='__main__':
    gen_tricks(23,31)
    print('all done')




