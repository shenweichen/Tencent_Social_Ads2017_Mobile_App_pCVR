
# coding: utf-8

# In[1]:

import os
import pickle
import pandas as pd
import numpy as np
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle


# In[2]:

def gen_global_index():
    train = pd.read_csv(raw_data_path+'train.csv')
    test = pd.read_csv(raw_data_path+'test.csv')
    all_data = train.append(test)
    all_data['global_index'] = np.arange(0,all_data.shape[0])
    train = all_data.iloc[0:train.shape[0],:]
    test = all_data.iloc[train.shape[0]:,:]
    dump_pickle(train,raw_data_path+'train.pkl')
    dump_pickle(test,raw_data_path+'test.pkl')


# In[3]:

def csv_pkl(csv_name_without_suffix,protocol=None):
    pkl_path = raw_data_path+csv_name_without_suffix +'.pkl'
    if not os.path.exists(pkl_path):
        print('generating '+pkl_path)
        data = pd.read_csv(raw_data_path+csv_name_without_suffix+'.csv')
        dump_pickle(data,pkl_path,protocol=protocol)
    else:
        print('found '+pkl_path)


# In[4]:

def gen_demo_result():
    test = pd.read_csv(raw_data_path+'test.csv')
    test = test[['instanceID','label']]
    test.rename(columns={'label':'prob'},inplace=True)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    test.to_csv(result_path+'demo_result.csv',index=False)


# In[7]:

if __name__ == '__main__':
    gen_global_index()
    train = load_pickle(raw_data_path+'train.pkl')
    train = train[train.clickTime>=17000000]#丢弃16号的数据
    dump_pickle(train,raw_data_path+'train.pkl')
    
    csv_pkl('ad')
    csv_pkl('position')
    csv_pkl('app_categories')
    csv_pkl('test')
    csv_pkl('user_app_actions')
    csv_pkl('user')
    csv_pkl('user_installedapps',protocol=4)
    
    gen_demo_result()
    
    if not os.path.exists(feature_data_path):
        os.mkdir(feature_data_path)
    if not os.path.exists(cache_pkl_path):
        os.mkdir(cache_pkl_path)


# In[ ]:



