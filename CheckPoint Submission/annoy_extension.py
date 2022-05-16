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
from pathlib import Path
import pickle
import os
import pandas as pd


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project Fast search on Small')
    netID = getpass.getuser()

    print('Reading ratings.csv and specifying schema')
    small_train_path = "hdfs:/user/" + netID + "/als_annoy_small_validation.parquet"
    small_val_path = "hdfs:/user/" + netID + "/als_annoy_small_test.parquet"
    small_test_path = "hdfs:/user/" + netID + "/als_annoy_small_train.parquet"

    train = pd.read_parquet(small_train_path)
    train.head()
    print(train.head())


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)