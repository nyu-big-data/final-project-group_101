#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.sql.functions import collect_list, explode, array_repeat, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import os

import itertools
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
    
def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    small_train_path = "hdfs:/user/" + netID + "/ratings_small_train.csv"
    small_val_path = "hdfs:/user/" + netID + "/ratings_small_val.csv"
    small_test_path = "hdfs:/user/" + netID + "/ratings_small_test.csv"
    
    ratings_small_train = spark.read.csv(small_train_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_val = spark.read.csv(small_val_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_test = spark.read.csv(small_test_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Give the dataframe a temporary view so we can run SQL queries
    ratings_small_train.createOrReplaceTempView('ratings_small_train')
    ratings_small_val.createOrReplaceTempView('ratings_small_val')
    ratings_small_test.createOrReplaceTempView('ratings_small_test')
    
    # Create Label(actual value) for validation set
    label_val = ratings_small_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_val = label_val.filter("userId is not null").select("label", "userId")
    label_val = label_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    # Create Label(actual value) for test set
    label_test = ratings_small_test.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_test = label_test.filter("userId is not null").select("label", "userId")
    label_test = label_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    
    
    rank_set = [50]
    maxIter = [10, 20]
    regParam = [0.001, 0.01]
    alpha = [5]
    s =[rank_set, maxIter, regParam, alpha]
    grid_set = np.array(list(itertools.product(*s)))
    
    cols = ["rank", "regParam", "alpha", "maxIter", "val_MAP", "val_NDCG", "test_MAP", "test_NDCG"]
    df = pd.DataFrame(columns=cols)
    
    # build the model
    for grid in grid_set:
        als = ALS(rank = grid[0], maxIter=grid[1], regParam=grid[2], alpha = grid[3], userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
        model = als.fit(ratings_small_train)
        
    
        # Validation Prediction
        predictions_val = model.recommendForUserSubset(label_val, 100)
        predictions_val.createOrReplaceTempView('predictions_val')
    
        predictions_val = predictions_val.select("userId", "recommendations.movieId").withColumnRenamed("movieId", "prediction")
        # predictions_val = predictions_val.withColumnRenamed("movieId", "prediction")
        dataset_val = label_val.join(predictions_val, label_val.userId == predictions_val.userId, 'inner').select("prediction", "label")
        dataset_val = dataset_val.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
        dataset_val = dataset_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
        
        
        # Test Prediction
        predictions_test = model.recommendForUserSubset(label_test, 100)
        predictions_test.createOrReplaceTempView('predictions_test')
    
        predictions_test = predictions_test.select("userId", "recommendations.movieId")
        predictions_test = predictions_test.withColumnRenamed("movieId", "prediction")
        dataset_test = label_test.join(predictions_test, label_test.userId == predictions_test.userId, 'inner').select("prediction", "label")
        dataset_test = dataset_test.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
        dataset_test = dataset_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))
        
        
        
        MAP_evaluator = RankingEvaluator(metricName='meanAveragePrecisionAtK', k = 100).setPredictionCol("prediction")
        NDCG_evaluator = RankingEvaluator(metricName = "ndcgAtK", k = 100).setPredictionCol("prediction")
        val_MAP = MAP_evaluator.evaluate(dataset_val)
        val_NDCG = NDCG_evaluator.evaluate(dataset_val)
        
        MAP_evaluator = RankingEvaluator(metricName='meanAveragePrecisionAtK', k = 100).setPredictionCol("prediction")
        NDCG_evaluator = RankingEvaluator(metricName = "ndcgAtK", k = 100).setPredictionCol("prediction")
        test_MAP = MAP_evaluator.evaluate(dataset_test)
        test_NDCG = NDCG_evaluator.evaluate(dataset_test)
        
        c = np.append(grid, np.array([val_MAP, val_NDCG, test_MAP, test_NDCG])).reshape(1, -1)
        df = df.append(pd.DataFrame(c, columns=cols))
    
    # print(df)
    spark.createDataFrame(df).show(truncate=False)

    
    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)