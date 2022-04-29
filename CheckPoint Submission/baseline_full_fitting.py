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


    print('Reading ratings.csv and specifying schema')
    full_train_path = "hdfs:/user/" + netID + "/ratings_full_train.csv"
    ratings_full_train = spark.read.csv(full_train_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    print("finish part 1")
    
    # Give the dataframe a temporary view so we can run SQL queries
    ratings_full_train.createOrReplaceTempView('ratings_full_train')
    print("finish part 2")


    # generate top 100 movie and convert to a list
    query_train = spark.sql('SELECT movieId, SUM(rating)/(COUNT(rating)+101) AS Utility_Score from ratings_full_train GROUP BY movieId ORDER BY Utility_Score DESC limit 100')
    top100 = query_train.select("movieId")
    print("finish part 3")
    top100_path = "hdfs:/user/" + netID + "/top100_full.csv"
    top100.createOrReplaceTempView('top100')
    top100.write.csv("top100_path")
    



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').config("spark.sql.broadcastTimeout", "36000").getOrCreate()
  
  
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
