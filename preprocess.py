from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sparse
import wget
import tarfile
import pip


import numpy as np
import pandas as pd


def encode_column(column):
    keys = column.unique()
    key_to_id = {key:idx for idx,key in enumerate(keys)}
    return key_to_id, np.array([key_to_id[x] for x in column]), len(keys)


def encode_df(anime_df):    
    anime_ids, anime_df['anime_id'], num_anime = encode_column(anime_df['anime_id'])
    user_ids, anime_df['user_id'], num_users = encode_column(anime_df['user_id'])
    return anime_df, num_users, num_anime, user_ids, anime_ids


def prep_anime(path):
    anime_ratings_df = pd.read_csv(path)
    anime_ratings = anime_ratings_df.loc[anime_ratings_df.rating != -1].reset_index()[['user_id','anime_id','rating']]
    anime_df, num_users, num_anime, user_ids, anime_ids = encode_df(anime_ratings)
    return anime_df, num_users, num_anime, user_ids, anime_ids


def train_test_anime(path):
    anime_df, num_users, num_anime, user_ids, anime_ids = prep_anime(path)
    sparse_item_user = sparse.csr_matrix((anime_df['rating'].values,(anime_df['user_id'].values, anime_df['anime_id'].values)),shape=(num_users, num_anime))
    sparse_user_item = sparse.csr_matrix((anime_df['rating'].values,(anime_df['anime_id'].values, anime_df['user_id'].values)),shape=(num_anime, num_users))
    data_conf = (sparse_item_user * alpha_val).astype('double')
    train_matrix, test_matrix = sklearn_train_test_split(data_conf)
    return train_matrix, test_matrix
    
    
def prep_lastfm():
    filename = wget.download('http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz')
    file = tarfile.open('lastfm-dataset-360K.tar.gz')
    file.extractall('./lastfm-dataset-360K')
    file.close()
    raw_data = pd.read_table('lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv')
    raw_data = raw_data.drop(raw_data.columns[1], axis=1)
    raw_data.columns = ['user', 'artist', 'plays']
    data = raw_data.dropna()
    data = data.copy()
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")
    data['user_id'] = data['user'].cat.codes
    data['artist_id'] = data['artist'].cat.codes
    return data
    
    
def prep_movielens():
    try:
        __import__('recommenders')
    except ImportError:
        pip.main(['install', 'recommenders'])       
    
    from recommenders.datasets import movielens
        
    data = movielens.load_pandas_df(
        size='100k',
        genres_col='genre',
        header=["userID", "itemID", "rating"]
        )
    return data
    
    
    
def prep_netflix(path):
    dataset = pd.read_csv(path,header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    dataset['Rating'] = dataset['Rating'].astype(float)
    df = pd.isnull(dataset['Rating'])
    df1 = pd.DataFrame(df)
    df2 = df1[df1['Rating']==True]
    df2 = df2.reset_index()
    df_nan = df2.copy()
    movie_np = []
    movie_id = 1

    for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
        temp = np.full((1,i-j-1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    last_record = np.full((1,len(dataset) - df_nan.iloc[-1, 0] - 1),movie_id)
    movie_np = np.append(movie_np, last_record)
    dataset = dataset[pd.notnull(dataset['Rating'])]

    dataset['Movie_Id'] = movie_np.astype(int)
    dataset['Cust_Id'] =dataset['Cust_Id'].astype(int)
    f = ['count','mean']
    dataset_movie_summary = dataset.groupby('Movie_Id')['Rating'].agg(f)
    dataset_movie_summary.index = dataset_movie_summary.index.map(int)
    movie_benchmark = round(dataset_movie_summary['count'].quantile(0.7),0)

    drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] < movie_benchmark].index
    dataset_cust_summary = dataset.groupby('Cust_Id')['Rating'].agg(f)
    dataset_cust_summary.index = dataset_cust_summary.index.map(int)
    cust_benchmark = round(dataset_cust_summary['count'].quantile(0.7),0)
    drop_cust_list = dataset_cust_summary[dataset_cust_summary['count'] < cust_benchmark].index
    dataset = dataset[~dataset['Movie_Id'].isin(drop_movie_list)]
    dataset = dataset[~dataset['Cust_Id'].isin(drop_cust_list)]
    return dataset 