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
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project BaseLine Model')

    small_train_path = "hdfs:/user/" + netID + "/ratings_small_train.csv"
    small_val_path = "hdfs:/user/" + netID + "/ratings_small_val.csv"
    small_train_test = "hdfs:/user/" + netID + "/ratings_small_test.csv"
    print('Reading ratings.csv and specifying schema')
    ratings_small_train = spark.read.csv(small_train_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_val = spark.read.csv(small_val_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_test = spark.read.csv(small_train_test, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Give the dataframe a temporary view so we can run SQL queries
    ratings_small_train.createOrReplaceTempView('ratings_small_train')
    ratings_small_val.createOrReplaceTempView('ratings_small_val')
    ratings_small_test.createOrReplaceTempView('ratings_small_test')
    

    # generate top 100 movie and convert to a list
    query_train = spark.sql('SELECT movieId, SUM(rating)/(COUNT(rating)+101) AS Utility_Score from ratings_small_train GROUP BY movieId ORDER BY Utility_Score DESC limit 100')
    top100 = query_train.select("movieId")
    top100 = top100.agg(collect_list("movieId"))
    top100 = top100.withColumnRenamed("collect_list(movieId)", "prediction")
    top100 = top100.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    
    
    
    # Validation
    # generate the required dataset for evaluate the performence using MAP
    temp_val = ratings_small_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    temp_val = temp_val.filter("userId is not null").select("label")
    temp_val = temp_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    dataset_val = temp_val.join(top100)
    
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    val_MAP = evaluator.evaluate(dataset_val)
    print("Validation Set Finished")
    
    
    # Test
    # generate the required dataset for evaluate the performence using MAP
    temp_test = ratings_small_test.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    temp_test = temp_test.filter("userId is not null").select("label")
    temp_test = temp_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    dataset_test = temp_test.join(top100)
    print("Test Set Finished")
    
    
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    test_MAP = evaluator.evaluate(dataset_test)
    
    
    print("Validation Set Performence with MAP: ", val_MAP)
    print("Test Set Performence with MAP: ", test_MAP)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
