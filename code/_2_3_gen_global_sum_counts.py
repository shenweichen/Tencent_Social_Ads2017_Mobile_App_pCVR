
# coding: utf-8

# In[1]:

import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle,dump_pickle,raw_data_path,feature_data_path
from feature_joint import addTime,addAd,addPosition,addAppCategories,addUserInfo


# In[2]:

def gen_ID_global_sum_count(last_day = 27,stats_features = ['positionID','creativeID','appID','adID','userID']):
    train = load_pickle(raw_data_path+'train.pkl')
    test = load_pickle(raw_data_path+'test.pkl')
    data = train.append(test)
    data = addTime(data)
    data = data[data.clickDay<=last_day]
    del train,test
    gc.collect()
    data = addAd(data)
    data = addPosition(data)
    data= addAppCategories(data)
    
    
    for feature in tqdm(stats_features):
        feature_path = feature_data_path+'global_count_'+feature+'_lastday'+str(last_day)+'.pkl'
        if os.path.exists(feature_path):
            print('found '+feature_path)
            #continue
        print('generating '+feature_path)
        feature_count_sum = pd.DataFrame(data.groupby(feature).size()).reset_index().rename(columns={0:feature+'_sum_count'})
        dump_pickle(feature_count_sum,feature_path)


# In[3]:

def add_global_count_sum(data,last_day =27 ,stats_features=['positionID','creativeID','appID','adID','userID']):
    """
    添加ID出现次数，根据ID_name拼接
    """
    for feature in tqdm(stats_features):
        feature_path = feature_data_path+'global_count_'+feature+'_lastday'+str(last_day)+'.pkl'
        if not os.path.exists(feature_path):
            gen_ID_global_sum_count([feature])
        feature_count_sum = load_pickle(feature_path)
        data = data.merge(feature_count_sum,'left',[feature])
    return data


# In[4]:

if __name__ =='__main__':
    gen_ID_global_sum_count(27)
    gen_ID_global_sum_count(31)
    print('all done')


# In[ ]:



