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

import itertools
import numpy as np
import pandas as pd
import time
    
def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project ALS Model on Small')


    print('Reading ratings.csv and specifying schema')
    small_train_path = "hdfs:/user/" + netID + "/ratings_small_train.csv"
    small_val_path = "hdfs:/user/" + netID + "/ratings_small_test.csv"
    
    ratings_small_train = spark.read.csv(small_train_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_val = spark.read.csv(small_val_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Give the dataframe a temporary view so we can run SQL queries
    ratings_small_train.createOrReplaceTempView('ratings_small_train')
    ratings_small_val.createOrReplaceTempView('ratings_small_val')
    
    # Create Label(actual value) for validation set
    label_val = ratings_small_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_val = label_val.filter("userId is not null").select("label", "userId")
    label_val = label_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    
    
    
    rank_set = 150
    maxIter = 20
    regParam = 1e-4
    alpha = 5
    grid =[rank_set, maxIter, regParam, alpha]
    
    
    
    
    # build the model
    
    start = time.time()
    als = ALS(rank = grid[0], maxIter=grid[1], regParam=grid[2], alpha = grid[3], userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(ratings_small_train)
        
    
    # Validation Prediction
    predictions_val = model.recommendForUserSubset(label_val, 100)
    predictions_val.createOrReplaceTempView('predictions_val')
    predictions_val = predictions_val.select("userId", "recommendations.movieId").withColumnRenamed("movieId", "prediction")
    
    # predictions_val = predictions_val.withColumnRenamed("movieId", "prediction")
    dataset_val = label_val.join(predictions_val, label_val.userId == predictions_val.userId, 'inner').select("prediction", "label")
    dataset_val = dataset_val.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_val = dataset_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    
    
    MAP_evaluator = RankingEvaluator(metricName='meanAveragePrecisionAtK', k = 100).setPredictionCol("prediction")
    NDCG_evaluator = RankingEvaluator(metricName = "ndcgAtK", k = 100).setPredictionCol("prediction")
    PREC_evaluator = RankingEvaluator(metricName = "precisionAtK", k = 100).setPredictionCol("prediction")
    val_MAP = MAP_evaluator.evaluate(dataset_val)
    val_NDCG = NDCG_evaluator.evaluate(dataset_val)
    val_PREC = PREC_evaluator.evaluate(dataset_val)
    
    end = time.time()
    print('{:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}'.format("rank", "maxIter", "regParam", "alpha", "val_MAP", "val_NDCG", "val_PREC"))
    print('{:15.2f} {:15.2f} {:15.6f} {:15.2f} {:15.6f} {:15.6f} {:15.6f}'.format(grid[0], grid[1], grid[2], grid[3], val_MAP, val_NDCG, val_PREC))
    
    print ("running time: ", end -start, "s")
    print("start: ", start)
    print("end: ", end)


    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').config("spark.sql.broadcastTimeout", "36000").getOrCreate()


    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)