
# coding: utf-8

# In[1]:

import os
import zipfile
import time
import pickle
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm



from feature_joint import addTime,addAd,addPosition,addAppCategories,addUserInfo
from _2_1_gen_user_click_features import add_user_day_click_count,add_user_day_hour_count,add_user_click_stats,add_user_day_click
from _2_2_gen_app_install_features import add_user_hist_install,add_user_start_installed_cateA
from _2_3_gen_global_sum_counts import add_global_count_sum
from _2_4_gen_tricks import add_tricks
from _2_5_gen_smooth_cvr import add_hist_cvr_smooth,add_smooth_pos_cvr
from _2_6_gen_ID_click_vectors import get_ConcatedTfidfVector_ID_user_clicks

from utils import load_pickle,dump_pickle,get_feature_value,feature_spearmanr,feature_target_spearmanr,addCrossFeature,calibration
from utils import raw_data_path,feature_data_path,cache_pkl_path,analyse


# In[2]:

def load_data(start_day=23,end_day=26,load_test=False):
    """
    读取基本表拼接后的数据
    test表load_test = True
    """
    if load_test ==True:
        trans_train_path = feature_data_path+'trans_test_'+str(start_day)+'_'+str(end_day)+'.pkl'
        raw_train_path = raw_data_path +'test.pkl'
    else:
        trans_train_path = feature_data_path+'trans_train_'+str(start_day)+'_'+str(end_day)+'.pkl'
        raw_train_path = raw_data_path +'train.pkl'

    if os.path.exists(trans_train_path):
        print('found '+trans_train_path)
        train = pickle.load(open(trans_train_path,'rb'))
    else:
        print('generating '+trans_train_path)
        train = load_pickle(raw_train_path)

        train = addTime(train)
        train = train[(train.clickDay>=start_day)&(train.clickDay<=end_day)]
        train = addAd(train)
        train = addPosition(train)
        train = addAppCategories(train)
        train = addUserInfo(train)
             
        dump_pickle(train,trans_train_path)
    return train


def merge_ID_vector(split_train_x,split_test_x,ID_name,last_day,concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby',],mode='local'):
    a = get_ConcatedTfidfVector_ID_user_clicks(ID_name,last_day,mode,concated_list=concated_list,drop_na=False)
    split_train_x = pd.merge(split_train_x,a,'left',ID_name)
    split_test_x = pd.merge(split_test_x,a,'left',ID_name)
    return split_train_x,split_test_x
    
def gen_online_data(train_start_day,train_end_day,test_day):
    
    train_x_path = cache_pkl_path +'online_train_x_'+str(train_start_day)+'_'+str(train_end_day)+'.pkl'
    test_x_path = cache_pkl_path + 'online_test_x_'+str(test_day)+'_'+str(test_day)+'.pkl'
    
    alpha = 0.647975342478
    beta = 34.83752176
    pos_na = alpha / (alpha + beta)
    
    if os.path.exists(train_x_path):
        print('found '+train_x_path)
        train_x  = load_pickle(train_x_path)
    else:
        print('generating '+train_x_path)
        train_x = load_data(train_start_day,train_end_day,False)
        train_x['age_cut']=pd.cut(train_x['age'],bins=[-1,0,18,25,35,45,55,65,np.inf],labels=False)
        
        #-----------trick-------------------------------
        train_x = add_tricks(train_x)
        #-----------intstall和action表相关----------------------
        print('adding install and actions...')
        train_x = add_user_start_installed_cateA(train_x)
        train_x = add_user_hist_install(train_x,'train')
         #-----------用户点击相关---------------------------- 
        print('adding user clicks...')
        train_x = add_user_day_click_count(train_x,['camgaignID','adID','sitesetID','appID',])
        train_x = add_user_day_hour_count(train_x,['camgaignID','adID','sitesetID','appID',])
        train_x = add_user_day_click(train_x)
        train_x = add_user_click_stats(train_x,)
         #------------转化率相关------------------------------
        print('adding conversions')
        train_x = add_smooth_pos_cvr(train_x,test_day)
        train_x = train_x.fillna({'positionID_cvr_smooth': pos_na})
        for cvr_key in [ 'creativeID', 'adID', 'appID','userID']:
            train_x = add_hist_cvr_smooth(train_x,cvr_key)

        #--------------其他------------------------------------
        train_x = add_global_count_sum(train_x,test_day,stats_features=['positionID', 'creativeID', 'appID', 'adID', 'userID'])
        
        
        dump_pickle(train_x,train_x_path)
    
    if os.path.exists(test_x_path):
        print('found '+test_x_path)
        test_x  = load_pickle(test_x_path)
    else:
        print('generating '+test_x_path)
        test_x = load_data(test_day,test_day,True)
        test_x['age_cut']=pd.cut(test_x['age'],bins=[-1,0,18,25,35,45,55,65,np.inf],labels=False)
        
        #-----------trick-------------------------------
        test_x = add_tricks(test_x)
        #-----------intstall和action表相关----------------------
        print('adding install and actions...')
        test_x = add_user_start_installed_cateA(test_x)
        test_x = add_user_hist_install(test_x,'test')
         #-----------用户点击相关---------------------------- 
        print('adding user clicks...')
        test_x = add_user_day_click_count(test_x,['camgaignID','adID','sitesetID','appID',])
        test_x = add_user_day_hour_count(test_x,['camgaignID','adID','sitesetID','appID',])
        test_x = add_user_day_click(test_x)
        test_x = add_user_click_stats(test_x,)
         #------------转化率相关------------------------------
        print('adding conversions')
        test_x = add_smooth_pos_cvr(test_x,test_day)
        test_x = test_x.fillna({'positionID_cvr_smooth': pos_na})
        for cvr_key in [ 'creativeID', 'adID', 'appID','userID']:
            test_x = add_hist_cvr_smooth(test_x,cvr_key)

        #--------------其他------------------------------------
        test_x = add_global_count_sum(test_x,test_day,stats_features=['positionID', 'creativeID', 'appID', 'adID', 'userID'])
       
        dump_pickle(test_x,test_x_path)
        
    train_x,test_x = merge_ID_vector(train_x,test_x,'advertiserID',last_day=test_day,concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby'])
    train_x,test_x = merge_ID_vector(train_x,test_x,'appID',last_day=test_day,concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby'])
    
    return train_x,test_x
  

def gen_offline_data(train_start_day,train_end_day,test_day,):
    
    train_x_path = cache_pkl_path +'offline_train_x_'+str(train_start_day)+'_'+str(train_end_day)+'.pkl'
    test_x_path = cache_pkl_path + 'offline_test_x_'+str(test_day)+'_'+str(test_day)+'.pkl'
    if os.path.exists(train_x_path) and os.path.exists(test_x_path):
        print('found offline data')
        train_x = load_pickle(train_x_path)
        test_x = load_pickle(test_x_path)
    else:
        alpha = 0.640792339811
        beta = 34.2999347427
        pos_na = alpha / (alpha + beta)
        
        print('generating offline data')
        train_x = load_data(train_start_day,test_day,False)
        train_x['age_cut']=pd.cut(train_x['age'],bins=[-1,0,18,25,35,45,55,65,np.inf],labels=False)
            
        #-----------trick-------------------------------
        train_x = add_tricks(train_x)
        #-----------intstall和action表相关----------------------
        print('adding install and actions...')
        train_x = add_user_start_installed_cateA(train_x)
        train_x = add_user_hist_install(train_x,)

         #-----------用户点击相关---------------------------- 
        print('adding user clicks...')
        train_x = add_user_day_click_count(train_x,['camgaignID','adID','sitesetID','appID',])
        train_x = add_user_day_hour_count(train_x,['camgaignID','adID','sitesetID','appID',])
        train_x = add_user_day_click(train_x)
        train_x = add_user_click_stats(train_x,)
         #------------转化率相关------------------------------
        print('adding conversions')
        train_x = add_smooth_pos_cvr(train_x,test_day)
        train_x = train_x.fillna({'positionID_cvr_smooth': pos_na})
        for cvr_key in [ 'creativeID', 'adID', 'appID','userID']:
            train_x = add_hist_cvr_smooth(train_x,cvr_key)

        #--------------其他------------------------------------
        train_x = add_global_count_sum(train_x,test_day,stats_features=['positionID', 'creativeID', 'appID', 'adID', 'userID'])
        #------------分割train和test--------------------------
        print('splitting train and test ...')
        test_x = train_x[train_x.clickDay==test_day]
        train_x = train_x[(train_x.clickDay>=train_start_day)&(train_x.clickDay<=train_end_day)]
        dump_pickle(train_x,train_x_path,4)
        dump_pickle(test_x,test_x_path,4)
    train_x,test_x = merge_ID_vector(train_x,test_x,'advertiserID',last_day=test_day,concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby'])
    train_x,test_x = merge_ID_vector(train_x,test_x,'appID',last_day=test_day,concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby'])
    return train_x,test_x


# In[3]:

if __name__ == '__main__':
    gen_online_data(25,29,31)
    #gen_online_data(25,25,31)

