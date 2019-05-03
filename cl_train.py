#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    $ spark-submit supervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
# TODO: you may need to add imports here


def main(spark, data_file, model_file):
    # Load the dataframe
    df = spark.read.parquet(data_file)
    df_sub = df.sample(True, 0.01)
    
    user_indexer  = StringIndexer(inputCol = "user_id", outputCol = "userNew", handleInvalid = "skip")
    track_indexer = StringIndexer(inputCol = "track_id", outputCol = "trackNew", handleInvalid = "skip")

    # ALS Model 
    als = ALS(maxIter=5, \
             userCol="userNew", itemCol="trackNew", ratingCol="count",\
             coldStartStrategy="drop")
    
    pipeline = Pipeline(stages = [user_indexer, track_indexer, als]) 

    paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.0, 0.001, 0.01, 0.1, 0.5, 1,10]) \
                                  .addGrid(als.alpha, [0.01, 0.1, 1, 5, 10]) \
                                  .addGrid(als.rank, [5, 10, 20, 50, 100, 500, 1000]) \
                                  .build()
    
    #cross vavildation 5-fold 
    crossval = CrossValidator(estimator = pipeline, estimatorParamMaps = paramGrid, \
				evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",\
                                predictionCol="prediction")(), numFolds = 5)

    model = crossval.fit(df_sub)
    model = model.bestModel 
    model.write().overwrite().save(model_file)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('cl_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
