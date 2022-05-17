import numpy as np
import pandas as pd
from time import time
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import lightfm
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import matplotlib.pyplot as plt
#Use getpass to obtain user netID
import getpass

print('Final Project Extension LightFM on Small dataset')

netID = getpass.getuser()

print('Reading ratings.csv and specifying schema')
small_train_path = "/home/" + netID + "/ratings_small_train.csv"
small_val_path = "/home/" + netID + "/ratings_small_val.csv"
small_test_path = "/home/" + netID + "/ratings_small_test.csv"

# Loading dataset from home directory
ratings_small_train = pd.read_csv(small_train_path, names = ('userId', 'movieId', 'rating', 'timestamp'))
ratings_small_val = pd.read_csv(small_val_path, names = ('userId', 'movieId', 'rating', 'timestamp', "median_timestamp"))
ratings_small_test = pd.read_csv(small_test_path, names = ('userId', 'movieId', 'rating', 'timestamp', "median_timestamp"))

print('Dropping irrelavant columns')
# drop median_timestamp & timestamp columns
ratings_small_val = ratings_small_val.drop(columns=['median_timestamp'])
ratings_small_test = ratings_small_test.drop(columns=['median_timestamp'])
ratings_small_train = ratings_small_train.drop(columns=['timestamp'])
ratings_small_val = ratings_small_val.drop(columns=['timestamp'])
ratings_small_test = ratings_small_test.drop(columns=['timestamp'])

print('Reassigning movie IDs to avoid dimension error')
# reassign movie IDs to avoid dimension error when fitting LightFM model
total_item_user = pd.concat([ratings_small_train, ratings_small_test, ratings_small_val]).drop_duplicates()
total_item_user = total_item_user.sort_values(['movieId'])
total_item_user['new_movieId'] = (total_item_user.groupby(['movieId'], sort=False).ngroup()+1)

print('Appending new_movieId to existing train, test, and validation sets')
# append new_movieId to existing train, test, and validation test sets
ratings_small_train = ratings_small_train.merge(total_item_user, on=['movieId','userId','rating'], how="left")
ratings_small_val = ratings_small_val.merge(total_item_user, on=['movieId','userId','rating'], how="left")
ratings_small_test = ratings_small_test.merge(total_item_user, on=['movieId','userId','rating'], how="left")

print('Dropping original movieId columns')
# drop original movieId column
ratings_small_train = ratings_small_train.drop(columns=['movieId'])
ratings_small_val = ratings_small_val.drop(columns=['movieId'])
ratings_small_test = ratings_small_test.drop(columns=['movieId'])

print("Adjusting dataset dimensions to avoid unmatched dimension error")
# adjust dataset dimensions to avoid error of unmatched dimensions
data = Dataset()
data.fit(users = np.unique(total_item_user["userId"]), items = np.unique(total_item_user["new_movieId"]))

print("building interactions")
# build interactions
interactions_train, weights_train = data.build_interactions([(ratings_small_train['userId'][i], 
                                                              ratings_small_train['new_movieId'][i],
                                                              ratings_small_train['rating'][i]) for i in range(ratings_small_train.shape[0])])
interactions_val, weights_val = data.build_interactions([(ratings_small_val['userId'][i],
                                                          ratings_small_val['new_movieId'][i], 
                                                          ratings_small_val['rating'][i]) for i in range(ratings_small_val.shape[0])])

# WARP model
print("start training WARP model")
t = time()
model = LightFM(loss='warp', learning_rate=0.05)
model = model.fit(interactions = interactions_train, sample_weight= weights_train, 
                  epochs = 10, verbose = False)
t_round = round(time()-t, 5)
val_precision = precision_at_k(model, interactions_val, k = 100).mean()

print("LightFM WARP model on small dataset")
print("Precision at k is:", val_precision)
print("Time spent is:", t_round)

# BPR model
print("start training BPR model")
t = time()
model = LightFM(loss='bpr', learning_rate=0.05)
model = model.fit(interactions = interactions_train, sample_weight= weights_train, 
                  epochs = 10, verbose = False)
t_round = round(time()-t, 5)
val_precision = precision_at_k(model, interactions_val, k = 100).mean()

print("LightFM BPR model on small dataset")
print("Precision at k is:", val_precision)
print("Time spent is:", t_round)
