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
    
    full_train_path = "hdfs:/user/" + netID + "/ratings_full_train.csv"
    full_val_path = "hdfs:/user/" + netID + "/ratings_full_val.csv"
    full_test_path = "hdfs:/user/" + netID + "/ratings_full_test.csv"
    top100_path = "hdfs:/user/" + netID + "/top100_full.csv"
    top100_NoD_path = "hdfs:/user/" + netID + "/top100_full_NoD.csv"
    top100_D_path = "hdfs:/user/" + netID + "/top100_full_D.csv"
    
    ratings_full_val = spark.read.csv(full_val_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_full_test = spark.read.csv(full_test_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    top100 = spark.read.csv(top100_path, schema='movieId float')
    top100_NoD = spark.read.csv(top100_NoD_path, schema='movieId float')
    top100_D = spark.read.csv(top100_D_path, schema='movieId float')
    print("finish part 1")
    
    # Give the dataframe a temporary view
    ratings_full_val.createOrReplaceTempView('ratings_full_val')
    ratings_full_test.createOrReplaceTempView('ratings_full_test')
    top100.createOrReplaceTempView('top100')
    top100_NoD.createOrReplaceTempView('top100_NoD')
    top100_D.createOrReplaceTempView('top100_D')
    print("finish part 2")
    
    # convert top 100 movie to a list for future evaluation
    top100 = top100.agg(collect_list("movieId"))
    top100 = top100.withColumnRenamed("collect_list(movieId)", "prediction")
    top100 = top100.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    
    top100_NoD = top100_NoD.agg(collect_list("movieId"))
    top100_NoD = top100_NoD.withColumnRenamed("collect_list(movieId)", "prediction")
    top100_NoD = top100_NoD.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    
    top100_D = top100_D.agg(collect_list("movieId"))
    top100_D = top100_D.withColumnRenamed("collect_list(movieId)", "prediction")
    top100_D = top100_D.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    

    # Validation
    # generate the required dataset for evaluate the performence using MAP
    temp_val = ratings_full_val.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    temp_val = temp_val.filter("userId is not null").select("label")
    temp_val = temp_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    dataset_val = temp_val.join(top100)
    dataset_val_NoD = temp_val.join(top100_NoD)
    dataset_val_D = temp_val.join(top100_D)
    
    # Evaluate validation set
    # evaluator = RankingEvaluator()
    # evaluator.setPredictionCol("prediction")
    # val_MAP = evaluator.evaluate(dataset_val)
    # print("Validation Set Performence with MAP: ", val_MAP)
    
    
    # MAP_evaluator = RankingEvaluator(metricName = "meanAveragePrecision").setPredictionCol("prediction")
    MAP_evaluator = RankingEvaluator(metricName = "meanAveragePrecisionAtK", k = 100).setPredictionCol("prediction")
    NDCG_evaluator = RankingEvaluator(metricName = "ndcgAtK", k = 100).setPredictionCol("prediction")

    # val_MAP = MAP_evaluator.evaluate(dataset_val)
    val_MAP = MAP_evaluator.evaluate(dataset_val)
    val_NDCG = NDCG_evaluator.evaluate(dataset_val)
    print("Validation Set Performence with MAP: ", val_MAP)
    print("Validation Set Performence with NDCG: ", val_NDCG)
    
    val_MAP_NoD = MAP_evaluator.evaluate(dataset_val_NoD)
    val_NDCG_NoD = NDCG_evaluator.evaluate(dataset_val_NoD)
    print("Validation Set Performence with MAP_NoD: ", val_MAP_NoD)
    print("Validation Set Performence with NDCG_NoD: ", val_NDCG_NoD)
    
    val_MAP_D = MAP_evaluator.evaluate(dataset_val_D)
    val_NDCG_D = NDCG_evaluator.evaluate(dataset_val_D)
    print("Validation Set Performence with MAP_D: ", val_MAP_D)
    print("Validation Set Performence with NDCG_D: ", val_NDCG_D)
    
    
    
    
    # Test
    # generate the required dataset for evaluate the performence using MAP
    temp_test = ratings_full_test.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    temp_test = temp_test.filter("userId is not null").select("label")
    temp_test = temp_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))
    dataset_test = temp_test.join(top100)
    dataset_test_NoD = temp_test.join(top100_NoD)
    dataset_test_D = temp_test.join(top100_D)
    
    # Evaluate test set
    # evaluator = RankingEvaluator()
    # evaluator.setPredictionCol("prediction")
    # test_MAP = evaluator.evaluate(dataset_test)
    # print("Test Set Performence with MAP:: ", test_MAP)
    
    # MAP_evaluator = RankingEvaluator(metricName = "meanAveragePrecision").setPredictionCol("prediction")
    MAP_evaluator = RankingEvaluator(metricName = "meanAveragePrecisionAtK", k = 100).setPredictionCol("prediction")
    NDCG_evaluator = RankingEvaluator(metricName = "ndcgAtK", k = 100).setPredictionCol("prediction")

    # val_MAP = MAP_evaluator.evaluate(dataset_val)
    test_MAP = MAP_evaluator.evaluate(dataset_test)
    test_NDCG = NDCG_evaluator.evaluate(dataset_test)
    print("Test Set Performence with MAP: ", test_MAP)
    print("Test Set Performence with NDCG: ", test_NDCG)
    
    test_MAP_NoD = MAP_evaluator.evaluate(dataset_test_NoD)
    test_NDCG_NoD = NDCG_evaluator.evaluate(dataset_test_NoD)
    print("Test Set Performence with MAP_NoD: ", test_MAP_NoD)
    print("Test Set Performence with NDCG_NoD: ", test_NDCG_NoD)
    
    test_MAP_D = MAP_evaluator.evaluate(dataset_test_D)
    test_NDCG_D = NDCG_evaluator.evaluate(dataset_test_D)
    print("Test Set Performence with MAP_D: ", test_MAP_D)
    print("Test Set Performence with NDCG_D: ", test_NDCG_D)

    
    
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').config("spark.sql.broadcastTimeout", "36000").getOrCreate()
  
  
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
    
    
    
    