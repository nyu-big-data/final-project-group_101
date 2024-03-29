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
    print('Final Project ALS Model on Small')


    print('Reading ratings.csv and specifying schema')
    full_train_path = "hdfs:/user/" + netID + "/ratings_full_train.csv"
    full_val_path = "hdfs:/user/" + netID + "/ratings_full_val.csv"
    full_test_path = "hdfs:/user/" + netID + "/ratings_full_test.csv"
    
    ratings_small_train = spark.read.csv(full_train_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_val = spark.read.csv(full_val_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    ratings_small_test = spark.read.csv(full_test_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

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
    
    
    
    rank_set = 200
    maxIter = 20
    regParam = 0.01
    alpha = 5
    grid =[rank_set, maxIter, regParam, alpha]
    
    
    # build the model
        
    als = ALS(rank = rank_set ,maxIter = maxIter,regParam = regParam,alpha = alpha, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(ratings_small_train)
    
    #train al dataset generation
    label_train = ratings_small_train.groupby("userId").agg(collect_list("movieId")).withColumnRenamed("collect_list(movieId)", "label")
    label_train = label_train.filter("userId is not null").select("label", "userId")
    label_train = label_train.withColumn('label', col('label').cast(ArrayType(DoubleType())))

    predictions_train = model.recommendForUserSubset(label_train, 100)
    predictions_train.createOrReplaceTempView('predictions_train')

    predictions_train = predictions_train.select("userId", "recommendations.movieId")
    predictions_train= predictions_train.withColumnRenamed("movieId", "prediction")
    dataset_train = label_train.join(predictions_train,label_train.userId == predictions_train.userId, 'right')
    dataset_train = dataset_train.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_train = dataset_train.withColumn('label', col('label').cast(ArrayType(DoubleType())))

    
    # Validation Prediction
    predictions_val = model.recommendForUserSubset(label_val, 100)
    predictions_val.createOrReplaceTempView('predictions_val')

    predictions_val = predictions_val.select("userId", "recommendations.movieId")
    predictions_val = predictions_val.withColumnRenamed("movieId", "prediction")
    dataset_val = label_val.join(predictions_val, label_val.userId == predictions_val.userId, 'right')
    dataset_val = dataset_val.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_val = dataset_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))



    '''
    MAP_evaluator = RankingEvaluator(metricName='meanAveragePrecisionAtK', k = 100).setPredictionCol("prediction")
    NDCG_evaluator = RankingEvaluator(metricName = "ndcgAtK", k = 100).setPredictionCol("prediction")
    val_MAP = MAP_evaluator.evaluate(dataset_val)
    val_NDCG = NDCG_evaluator.evaluate(dataset_val)
    
    
    print('{:>15} {:>15} {:>15} {:>15} {:>15} {:>15}'.format("rank", "regParam", "alpha", "maxIter", "val_MAP", "val_NDCG"))
    print('{:15.2f} {:15.2f} {:15.6f} {:15.2f} {:15.6f} {:15.6f}'.format(grid[0], grid[1], grid[2], grid[3], val_MAP, val_NDCG))
    '''
    
    
    # Test Prediction
    predictions_test = model.recommendForUserSubset(label_test, 100)
    predictions_test.createOrReplaceTempView('predictions_test')

    predictions_test = predictions_test.select("userId", "recommendations.movieId")
    predictions_test = predictions_test.withColumnRenamed("movieId", "prediction")
    dataset_test = label_test.join(predictions_test, label_test.userId == predictions_test.userId, 'right')
    dataset_test = dataset_test.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_test = dataset_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))


    # generate item vector and user vector

    user_vec = model.userFactors.toDF('id','features')
    item_vec = model.itemFactors.toDF('id','features')
    

    als_annoy_user_path = "hdfs:/user/" + netID + "/user_vec_large.parquet"
    als_annoy_item_path = "hdfs:/user/" + netID + "/item_vec_large.parquet"


    user_vec.write.parquet(als_annoy_user_path)
    item_vec.write.parquet(als_annoy_item_path )
    
    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)