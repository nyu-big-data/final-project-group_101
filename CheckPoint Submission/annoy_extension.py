#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
pip install pyspark
pip install annoy
pip install pyarrow

import getpass
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, explode, array_repeat, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pathlib import Path
import pickle
import os
import pandas as pd
import pickle
import os
import time
import numpy as np
from tqdm import tqdm_notebook as tqdm 
import matplotlib.pyplot as plt 
#pip-install annoy
#from annoy import AnnoyIndex
#pip install pyarrow


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project Fast search on Small')
    netID = getpass.getuser()

    print('Reading ratings.csv and specifying schema')
    #small_train_path = "hdfs:/user/" + netID + "/als_annoy_small_validation.parquet"
    #small_val_path = "hdfs:/user/" + netID + "/als_annoy_small_test.parquet"
    #small_test_path = "hdfs:/user/" + netID + "/als_annoy_small_train.parquet"

    user_vector_path = "hdfs:/user/" + netID + "/user_vec.parquet"
    item_vector_path = "hdfs:/user/" + netID + "/item_vec.parquet"

    #train =spark.read.parquet(small_train_path)
    #val=spark.read.parquet(small_val_path)
    #test =spark.read.parquet(small_test_path)
    user =spark.read.parquet(user_vector_path)
    item =spark.read.parquet(item_vector_path)

    user.createOrReplaceTempView("user")
    item.createOrReplaceTempView("item")

    #train.createOrReplaceTempView("train_table")
    #val.createOrReplaceTempView("val_table")
    #test.createOrReplaceTempView("test_table")

    spark.sql("select id,features from user").show()
    spark.sql("select id,features from item").show()

    useru = spark.sql("select id,features from user").toDF('id','features')
    useru = useru.toPandas()


    itemu = spark.sql("select id,features from item").toDF('id','features')
    itemu = itemu.toPandas()


    #trainu = spark.sql("select label,userId,prediction from train_table").toDF('label','userId','prediction')
    #trainu = trainu.toPandas()

    #valu = spark.sql("select label,userId,prediction from val_table").toDF('label','userId','prediction')
    #valu = valu.toPandas()

    #testu = spark.sql("select label,userId,prediction from test_table").toDF('label','userId','prediction')
    #testu = testu.toPandas()

    user_vec = useru['features'].values
    item_vec = itemu['features'].values

    #Brute search

    def find_nearest_exhaustive(data, queries, k): #top-k recommendat 
        if len(data.shape) == 1:
            data = np.array([x for x in data]) 
        n_items = data.shape[0]
        n_feat = data.shape[1]
        n_queries = len(queries)

        def single_query(query):
            start = time.time()
            if type(query) is not np.ndarray:
                query = np.array(query)
            res = np.argsort(-data.dot(query))[:k] 
            interval = time.time() - start
            return interval, res

        times = []
        results = []
        for i in tqdm(range(n_queries)): #tqdm for loop visualization
            interval, res = single_query(queries[i]) 
            times.append(interval) 
            results.append(res)

        mean_time = sum(times) / len(times) 

        print('Exhaustive Brute-force Search\n')
        print('Mean Query Search: %.6f' % mean_time) 
        return mean_time, results

    bf_mean_time, bf_results = find_nearest_exhaustive(item_vec,user_vec,10)
    print(bf_results)


   
    ## Fast search with annoy

    f = len(item_vec[0])
    t = AnnoyIndex(f, metric='dot') 
    for i in range(len(item_vec)):
        t.add_item(i, item_vec[i])
    t.build(10)

    def wrap_with(obj, method, mapping): 
        '''
        obj: the model that can respond to the query
        method: the name of the query method
        mapping: what input be mapped
        '''
        get_map = lambda x: [x[mapping[i]] for i in range(len(mapping))]
        def wrapped(*args, **kwrds):
            return obj.__getattribute__(method)(*get_map(args)) 
        return wrapped


    def find_nearest_algo(data, queries, true_label, model_wrapped, k,extra_para): 
        if len(data.shape) == 1:
            data = np.array([x for x in data])

        n_items = data.shape[0]
        n_feat = data.shape[1]
        n_queries = len(queries)

        def single_query(query):
            start = time.time()
            res = model_wrapped(query, k, extra_para) 
            interval = time.time() - start
            return interval, res

        def get_recall(predict, truth):
            return len([x for x in predict if x in truth]) / len(truth)

        times = []
        recalls = []

        for i in tqdm(range(n_queries)):
            interval, res = single_query(queries[i]) 
            recall = get_recall(res, true_label[i]) 
            times.append(interval) 
            recalls.append(recall)

        mean_time = sum(times) / len(times) 
        mean_recall = sum(recalls) / len(recalls)
        print('-' * 26)
        print('Mean Query Search Time: %.6f' % mean_time) 
        print('Mean Recall: %.6f' % mean_recall)
        return mean_time, mean_recall

    wrapped = wrap_with(t, 'get_nns_by_vector', [0, 1, 2])
    find_nearest_algo(item_vec,user_vec, bf_results, wrapped, 10, 100) 


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)