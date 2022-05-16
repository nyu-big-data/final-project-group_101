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

    # generate top 100 movie without dumping variable and convert to a list
    query_train_NoD = spark.sql('SELECT movieId, SUM(rating)/(COUNT(rating)) AS Utility_Score from ratings_full_train GROUP BY movieId ORDER BY Utility_Score DESC limit 100')
    top100_NoD = query_train_NoD.select("movieId")
    print("finish part 3")
    top100_NoD_path = "hdfs:/user/" + netID + "/top100_full_NoD.csv"
    top100_NoD.createOrReplaceTempView('top100_NoD')
    top100_NoD.write.csv(top100_NoD_path)
    
    
    # generate top 100 movie with dumping = 101 and convert to a list
    query_train = spark.sql('SELECT movieId, SUM(rating)/(COUNT(rating)+101) AS Utility_Score from ratings_full_train GROUP BY movieId ORDER BY Utility_Score DESC limit 100')
    top100 = query_train.select("movieId")
    print("finish part 4")
    top100_path = "hdfs:/user/" + netID + "/top100_full.csv"
    top100.createOrReplaceTempView('top100')
    top100.write.csv(top100_path)
    
    # generate top 100 movie with dumping = 10100 and convert to a list
    query_train_D = spark.sql('SELECT movieId, SUM(rating)/(COUNT(rating)+10100) AS Utility_Score from ratings_full_train GROUP BY movieId ORDER BY Utility_Score DESC limit 100')
    top100_D = query_train_D.select("movieId")
    print("finish part 5")
    top100_D_path = "hdfs:/user/" + netID + "/top100_full_D.csv"
    top100_D.createOrReplaceTempView('top100_D')
    top100_D.write.csv(top100_D_path)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').config("spark.sql.broadcastTimeout", "36000").getOrCreate()
  
  
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
