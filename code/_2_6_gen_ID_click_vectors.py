
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


# # 根据点击量计算描述向量(advertiserID和appID有用)

# In[2]:

def gen_CountVector_ID_user_clicks(ID_name,last_day=27,ID_describe_feature_names=['age_cut','gender','education','marriageStatus','haveBaby',],drop_na = False):
    """
    生成根据train和test表计算的ID_name计数描述向量，可以进行其他后处理
    拼接键[ID_name]
    """
    train = load_pickle(raw_data_path+'train.pkl')
    test = load_pickle(raw_data_path+'test.pkl')
    data = train.append(test)
    data = addTime(data)
    data = data[data.clickDay<=last_day]
    data = addAd(data)
    data = addPosition(data)
    data = addAppCategories(data)
    data = data[['userID',ID_name]]
    user_info = pd.read_csv(raw_data_path+'user.csv')
    
    user_info['age_cut']=pd.cut(user_info['age'],bins=[-1,0,18,25,35,45,55,np.inf],labels=False)
    user_info.loc[user_info.education==7,'education'] = 6
    
    user_info['hometown_province'] = user_info['hometown'].apply(lambda x: x//100)
    user_info['residence_province'] = user_info['residence'].apply(lambda x: x//100)
    
    for feature in tqdm(ID_describe_feature_names):
        feature_path = feature_data_path +'CountVector_'+ID_name+'_user_clicks_'+feature+'_lastday'+str(last_day)+'.pkl'
        if drop_na:
            feature_path += '.no_na'
        if os.path.exists(feature_path):
            print('found '+feature_path)
            continue
        print('generating '+feature_path)
        prefix_name = ID_name+'_user_clicks_'+feature
        sub_user_info =pd.get_dummies(user_info[['userID',feature]],columns=[feature],prefix=prefix_name)
        if drop_na:
            sub_user_info.drop([prefix_name+'_0'],axis=1,inplace=True)
        data = pd.merge(data,sub_user_info,'left','userID')
        dummy_features= sub_user_info.columns.tolist()
        dummy_features.remove('userID')
        ID_describe_feature = data[[ID_name]+dummy_features].groupby([ID_name],as_index=False).sum()
        data.drop(dummy_features,axis=1,inplace=True)
        dump_pickle(ID_describe_feature,feature_path)

def get_ConcatedTfidfVector_ID_user_clicks(ID_name,last_day,mode='local',concated_list=['age_cut','gender','education','marriageStatus','haveBaby',],drop_na=False,norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
    """
    使用默认的local模式效果稍微好一些
    测试过advertiserID,camgaignID,adID,creativeID,appID,appCategory,cate_A,appPlatform,positionType
    adver效果较好,appID效果其次,然后是appCategory，其他都不好
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_vec = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    concated_tfidf_vec = None

    for feature in tqdm(concated_list):
        feature_path = feature_data_path +'CountVector_'+ID_name+'_user_clicks_'+feature+'_lastday'+str(last_day)+'.pkl'
        if drop_na:
            feature_path += '.no_na'
        if not os.path.exists(feature_path):
            gen_CountVector_ID_user_clicks(ID_name)
        count_vec = load_pickle(feature_path)
        if mode == 'local':
            count_vec.set_index(ID_name,inplace=True)
            vec_columns = count_vec.columns
            local_tfidf_vec = tfidf_vec.fit_transform(count_vec).todense()
            local_tfidf_vec = pd.DataFrame(local_tfidf_vec,columns=vec_columns,index=count_vec.index).reset_index()
        elif mode=='global':
            local_tfidf_vec = count_vec
            
        if concated_tfidf_vec is None:
            concated_tfidf_vec = local_tfidf_vec
        else:
            concated_tfidf_vec = pd.merge(concated_tfidf_vec,local_tfidf_vec,'left',ID_name)
    if mode == 'global':
        concated_tfidf_vec.set_index(ID_name,inplace=True)
        vec_columns = concated_tfidf_vec.columns
        global_concated_tfidf_vec = tfidf_vec.fit_transform(concated_tfidf_vec).todense()
        global_concated_tfidf_vec = pd.DataFrame(global_concated_tfidf_vec,columns = vec_columns,index=concated_tfidf_vec.index)
        concated_tfidf_vec = global_concated_tfidf_vec.reset_index()
    return concated_tfidf_vec


# In[3]:

if __name__ == '__main__':

    gen_CountVector_ID_user_clicks('advertiserID',31)
    gen_CountVector_ID_user_clicks('appID',31)
    gen_CountVector_ID_user_clicks('advertiserID',27)
    gen_CountVector_ID_user_clicks('appID',27)
    print('all done')


# In[ ]:



