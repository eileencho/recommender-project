'''Usage:

    $ spark-submit cl_train_pipe.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/ec3636/step_model

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

def main(spark, data_file, val_file, model_file):
    # Load the dataframe
    df = spark.read.parquet(data_file)
   # df = df.sample(True, 0.01)
    val_df = spark.read.parquet(val_file)
   # val_df = df.sample(True, 0.01) 
    
    user_indexer  = StringIndexer(inputCol = "user_id", outputCol = "userNew", handleInvalid = "skip")
    track_indexer = StringIndexer(inputCol = "track_id", outputCol = "trackNew", handleInvalid = "skip")
    
 #   user_indexed = user_indexer.fit(df)
 #   df =user_indexed.transform(df)
 #   track_indexed = track_indexer.fit(df)
 #   df =track_indexed.transform(df)
 #   val_df = user_indexed.transform(val_df)
 #   val_df = track_indexed.transform(val_df)

    # ALS Model 
    als = ALS(maxIter=5, \
             userCol="userNew", itemCol="trackNew", ratingCol="count",\
             coldStartStrategy="drop")

    # Pipeline
    pipeline = Pipeline(stages = [user_indexer, track_indexer, als]) 
    
    #paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.1, 1]).build()
                                 # .addGrid(als.alpha, [0.01, 0.1, 1, 5, 10]) \
                                 # .addGrid(als.rank, [5, 10, 20, 50, 100, 500, 1000]) \
    RegParam = [0.001, 0.01, 0.1, 1, 10]
    Alpha = [0.1, 1,5,10, 100]
    Rank = [5,10,50,100,1000]

    evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "count", predictionCol = "prediction")

    RMSE = {}
    count = 0
    for i in RegParam:
        for j in Alpha:
            for k in Rank:
                  als = ALS(maxIter=5, regParam = i, alpha = j, rank = k, \
                            userCol="userNew", itemCol="trackNew", ratingCol="count",\
                            coldStartStrategy="drop")
                  model = pipeline.fit(df)
                  val_predictions = model.transform(val_df)
                  rmse = evaluator.evaluate(val_predictions)
                  RMSE[rmse] = model
                  count += 1
                  print(count)
                  print(rmse)

    best_RMSE = min(list(RMSE.keys()))
    bestmodel = RMSE[best_RMSE]
    bestmodel.write().overwrite().save(model_file)

    print("model is complete... go sleep")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('cl_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    #validation file

    val_file = sys.argv[2]

    # And the location to store the trained model
    model_file = sys.argv[3]



    # Call our main routine
    main(spark, data_file, val_file, model_file)
