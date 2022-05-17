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
    
def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project ALS Model on Full')


    print('Reading ratings.csv and specifying schema')
    full_train_path = "ratings_full_train.csv"
    full_val_path = "ratings_full_val.csv"
    
    ratings_full_train = spark.read.csv(full_train_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_full_val = spark.read.csv(full_val_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Give the dataframe a temporary view so we can run SQL queries
    ratings_full_train.createOrReplaceTempView('ratings_full_train')
    ratings_full_val.createOrReplaceTempView('ratings_full_val')
    
    # Create Label(actual value) for validation set
    label_val = ratings_full_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_val = label_val.filter("userId is not null").select("label", "userId")
    label_val = label_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    
    
    
    rank_set = [500]
    maxIter = [10]
    regParam = [0.00001]
    alpha = [5]
    grid =[rank_set, maxIter, regParam, alpha]
    
    
    # build the model
        
    als = ALS(rank = 100,maxIter = 10,regParm = 0.1,alpha = 10, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(ratings_full_train)
        
    
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
    val_MAP = MAP_evaluator.evaluate(dataset_val)
    val_NDCG = NDCG_evaluator.evaluate(dataset_val)
    
    
    print('{:>15} {:>15} {:>15} {:>15} {:>15} {:>15}'.format("rank", "regParam", "alpha", "maxIter", "val_MAP", "val_NDCG"))
    print('{:15.2f} {:15.2f} {:15.6f} {:15.2f} {:15.6f} {:15.6f}'.format(grid[0], grid[1], grid[2], grid[3], val_MAP, val_NDCG))


    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()


    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)