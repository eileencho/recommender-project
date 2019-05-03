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

def main(spark, data_file, val_file, model_file):
    # Load the dataframe
    df = spark.read.parquet(data_file)
    df = df.sample(True, 0.01)
    val_df = spark.read.parquet(val_file)
    val_df = df.sample(True, 0.01) 
    
    user_indexer  = StringIndexer(inputCol = "user_id", outputCol = "userNew", handleInvalid = "skip")
    track_indexer = StringIndexer(inputCol = "track_id", outputCol = "trackNew", handleInvalid = "skip")
    
    df = user_indexer.fit(df).transform(df)
    df = item_indexer.fit(df).transform(df)
    val_df = user_indexer.transform(val_df)
    val_df = item_indexer.transform(val_df)

    # # ALS Model 
    # als = ALS(maxIter=5, alpha = 1, regParam = 1, rank = 10,  \
    #          userCol="userNew", itemCol="trackNew", ratingCol="count",\
    #          coldStartStrategy="drop")
    
    # model = als.fit(df) 
    
    # val_predictions = model.transform(val_df)
    
    # evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "count", predictionCol = "prediction")
    # rmse = evaluator.evaluate(val_predictions)
    # print("According to Tin, the root mean sqare error = " + str(rmse))

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
