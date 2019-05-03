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
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
# TODO: you may need to add imports here


def main(spark, data_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    # Load the dataframe
    df = spark.read.parquet(data_file)

    # Give the dataframe a temporary view so we can run SQL queries
    df.createOrReplaceTempView(df)
    
    # Select out the 20 attribute columns labeled mfcc_00, mfcc_01, ..., mfcc_19
    mfcc = ", ".join(["mfcc_0{}".format(i) for i in range(10)]+["mfcc_{}".format(i) for i in range(10,20)])
    data_table = spark.sql("SELECT {} FROM df".format(mfcc))
    data_table.createOrReplaceTempView(data_table)
    
    # Encode the genre field as a target label using a StringIndexer and store the result as label.
    labels = spark.sql("SELECT genre FROM df")
    indexer = StringIndexer() #inputCol="category", outputCol="categoryIndex"
    label = indexer.fit(labels).transform(labels)

    # Define a multi-class logistic regression classifier
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    # Optimize the hyper-parameters (elastic net parameter and regularization weight) of 
    #your model by 5-fold cross-validation on the training set. 
    #Use at least 5 distinct values for each parameter in your grid.
    lrModel = lr.fit(training)
    
    #  Combine the entire process into a Pipeline object. 
    # Once the model pipeline has been fit, save it to the provided model filename.
    model_file = model_file + "/lrmodel"
    lr.Model.save(model_file)
    ###

    pass




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
