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
    df.createOrReplaceTempView("df")
    val_df = spark.read.parquet(val_file)
    val_df.createOrReplaceTempView("val_df")
    #grab only the users present in validation sample
    df = spark.sql("SELECT * FROM df WHERE user_id IN (SELECT user_id FROM val_df)")
    #create and store indexer info
    user_indexer  = StringIndexer(inputCol = "user_id", outputCol = "userNew", handleInvalid = "skip")
    track_indexer = StringIndexer(inputCol = "track_id", outputCol = "trackNew", handleInvalid = "skip")
    pipeline = Pipeline(stages = [user_indexer, track_indexer]) 
    indexers = pipeline.fit(df)
    indexers.write().overwrite().save("./final/indexers")

    #transform
    df = indexers.transform(df).cache()
    val_df = indexers.transform(val_df).select(["userNew","trackNew"])

    groundTruth = val_df.groupby("userNew").agg(F.collect_list("trackNew").alias("truth")).cache()
    print("created ground truth df")
    RegParam = [0.0001, 0.1, 10] # 0.1, 1, 10]
    Alpha = [0.001, 0.01, 15 , 100]#5,10, 100]
    Rank = [10,50,100]

    PRECISIONS = {}
    count = 0
    total = len(RegParam)*len(Alpha)*len(Rank)
    for i in RegParam:
        for j in Alpha:
            for k in Rank:
                print(f"regParam: {i}, Alpha: {j}, Rank: {k}")
                als = ALS(maxIter=5, regParam = i, alpha = j, rank = k, \
                          userCol="userNew", itemCol="trackNew", ratingCol="count",\
                          coldStartStrategy="drop",implicitPrefs=True)
                alsmodel = als.fit(df)
                print("fit model")
                rec = alsmodel.recommendForAllUsers(500)
                print("got recs")
                predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew, 'inner')
                print("start mapping...")
                scoreAndLabels = predictions.select('recommendations.trackNew','truth').rdd.map(tuple)
                print("scoring...")
                metrics = RankingMetrics(scoreAndLabels)
                print("precision...")
                precision = metrics.precisionAt(500)
                PRECISIONS[precision] = alsmodel
                count += 1
                print(f"finished {count} of {total}")
                print(precision)
        	#print(f"count: {count}, regParam: {i}, alpha: {j}, rank: {k}, PRECISIONS: {precision}")


    best_precision = max(list(PRECISIONS.keys()))
    bestmodel = PRECISIONS[best_precision]
    bestmodel.write().overwrite().save(model_file)
    print(f"Best precision: {best_precision}, with regParam: {bestmodel.getRegParam()}, alpha: {bestmodel.getAlpha()}, rank: {bestmodel.getRank()}")
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
