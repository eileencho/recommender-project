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
from pyspark.sql import functions as F

from pyspark.mllib.evaluation import RankingMetrics

def main(spark, data_file, val_file, model_file):
    # Load the dataframe
    df = spark.read.parquet(data_file)
    df = df.sample(True, 0.01)
    val_df = spark.read.parquet(val_file)
    val_df = df.sample(True, 0.01) 
    
    user_indexer  = StringIndexer(inputCol = "user_id", outputCol = "userNew", handleInvalid = "skip")
    track_indexer = StringIndexer(inputCol = "track_id", outputCol = "trackNew", handleInvalid = "skip")
    pipeline = Pipeline(stages = [user_indexer, track_indexer]) 
    indexers = pipeline.fit(df)
    df = indexers.transform(df)
    val_df = indexers.transform(val_df).select(["userNew","trackNew"])
    groundTruth = val_df.groupby("userNew").agg(F.collect_list("trackNew").alias("truth"))
    print("created ground truth df")
    RegParam = [0.001, 0.01] # 0.1, 1, 10]
    Alpha = [0.1, 1]#5,10, 100]
    Rank = [5,10] #50,100,1000]
    sc = spark.sparkContext
    PRECISIONS = {}
    count = 0
    for i in RegParam:
        for j in Alpha:
            for k in Rank:
                print(f"i: {i}, j: {j}, k: {k}")
                als = ALS(maxIter=5, regParam = i, alpha = j, rank = k, \
                          userCol="userNew", itemCol="trackNew", ratingCol="count",\
                          coldStartStrategy="drop")
                alsmodel = als.fit(df)
                print("fit model")
                #val_predictions = model.transform(val_df)
                #alsmodel = model.stages[-1]
                rec = alsmodel.recommendForAllUsers(500)
                print("got recs")
                predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew, 'inner')
                print("start mapping...")
                #predictions.show(10)
                #print(rec.show(10))
                #scoreAndLabels = val_predictions.rdd
                #sc = spark.sparkContext
                #scoreAndLabels = sc.parallelize(scoreAndLabels)
                scoreAndLabels = predictions.select('recommendations','truth').rdd.map(tuple)
                print("scoring")
                metrics = RankingMetrics(scoreAndLabels)
                precision = metrics.precisionAt(500)
                PRECISIONS[precision] = model
                count += 1
                print(count)
                print(precision)
        	#print(f"count: {count}, regParam: {i}, alpha: {j}, rank: {k}, PRECISIONS: {precision}")


    best_precision = min(list(PRECISIONS.keys()))
    bestmodel = PRECISIONS[best_precision]
    bestmodel.write().overwrite().save(model_file)
    print(f"Best precision: {best_precision}, with regParam: {bestmodel.getregParam()}, alpha: {bestmodel.getAlpha()}, rank: {bestmodel.getRank()}")
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
