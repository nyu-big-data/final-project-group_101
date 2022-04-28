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

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project BaseLine Model')


    print('Reading ratings.csv and specifying schema')
    ratings_small_train = spark.read.csv('hdfs:/user/jz5246/ratings_small_train.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_val = spark.read.csv('hdfs:/user/jz5246/ratings_small_val.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_test = spark.read.csv('hdfs:/user/jz5246/ratings_small_test.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Give the dataframe a temporary view so we can run SQL queries
    ratings_small_train.createOrReplaceTempView('ratings_small_train')
    ratings_small_val.createOrReplaceTempView('ratings_small_val')
    ratings_small_test.createOrReplaceTempView('ratings_small_test')
    
    
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
    model = als.fit(ratings_small_train)
    
    # Validation
    predictions_val = model.transform(ratings_small_val)
    predictions_val.createOrReplaceTempView('predictions_val')
    predictions_val = spark.sql("SELECT userId, movieId FROM (SELECT userId, movieId, Rank() over (Partition BY userId ORDER BY prediction DESC ) AS Rank FROM predictions_val) T1 WHERE Rank <= 100")
    
    predictions_val = predictions_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "prediction")
    predictions_val = predictions_val.filter("userId is not null")
    predictions_val = predictions_val.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    
    
    label_val = ratings_small_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_val = label_val.filter("userId is not null").select("label", "userId")
    label_val = label_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    dataset_val = label_val.join(predictions_val, label_val.userId == predictions_val.userId, 'inner').select("prediction", "label")
    
    
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    val_MAE = evaluator.evaluate(dataset_val)
    print("Validation Performence with MAE: ", val_MAE)
    
    
    # Test
    predictions_test = model.transform(ratings_small_test)
    predictions_test.createOrReplaceTempView('predictions_test')
    predictions_test = spark.sql("SELECT userId, movieId FROM (SELECT userId, movieId, Rank() over (Partition BY userId ORDER BY prediction DESC ) AS Rank FROM predictions_test) T1 WHERE Rank <= 100")
    
    predictions_test = predictions_test.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "prediction")
    predictions_test = predictions_test.filter("userId is not null")
    predictions_test = predictions_test.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    
    
    label_test = ratings_small_test.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_test = label_test.filter("userId is not null").select("label", "userId")
    label_test = label_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    dataset_test = label_test.join(predictions_test, label_test.userId == predictions_test.userId, 'inner').select("prediction", "label")
    
    
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    test_MAE = evaluator.evaluate(dataset_test)
    print("Test Performence with MAE: ", test_MAE)




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)