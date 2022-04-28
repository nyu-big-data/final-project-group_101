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

    
    
def main(spark, netID):
    print('Final Project BaseLine Model')

    print('Reading ratings.csv and specifying schema')
    ratings_full_val = spark.read.csv('hdfs:/user/jz5246/ratings_full_val.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_full_test = spark.read.csv('hdfs:/user/jz5246/ratings_full_test.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    top100 = spark.read.csv("hdfs:/user/jz5246/top100_full.csv", schema='movieId float')
    print("finish part 1")
    
    # Give the dataframe a temporary view
    ratings_full_val.createOrReplaceTempView('ratings_full_val')
    ratings_full_test.createOrReplaceTempView('ratings_full_test')
    top100.createOrReplaceTempView('top100')
    print("finish part 2")
    
    # convert top 100 movie to a list for future evaluation
    top100 = top100.agg(collect_list("movieId"))
    top100 = top100.withColumnRenamed("collect_list(movieId)", "label")
    top100 = top100.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    

    # Validation
    # generate the required dataset for evaluate the performence using MAP
    temp_val = ratings_full_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "prediction")
    temp_val = temp_val.filter("userId is not null").select("prediction")
    temp_val = temp_val.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_val = temp_val.join(top100)
    
    # Evaluate validation set
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    val_MAE = evaluator.evaluate(dataset_val)
    print("Validation Set Finished: ", val_MAE)
    
    # Test
    # generate the required dataset for evaluate the performence using MAP
    temp_test = ratings_full_test.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "prediction")
    temp_test = temp_test.filter("userId is not null").select("prediction")
    temp_test = temp_test.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_test = temp_test.join(top100)
    
    # Evaluate test set
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    test_MAE = evaluator.evaluate(dataset_test)
    print("Test Set Finished: ", test_MAE)

    
    
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').config("spark.sql.broadcastTimeout", "36000").getOrCreate()
  
  
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
    
    
    
    