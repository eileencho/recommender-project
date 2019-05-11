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
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.mllib.evaluation import RankingMetrics

def main(spark, data_file, val_file, model_file1, model_file2, model_file3):
    # Load the dataframe    
    table = pq.read_table(data_file)
    
    #default compression is 'snappy'; create representations for alternate log compressions
    pq.write_table(table, "./df.parquet.gzip", compression='gzip')
    pq.write_table(table, "./df.parquet.brotli", compression='brotli')
    pq.write_table(table, "./df.parquet.none", compression='None')
    
    #read each file and take a sample to generate a model
    df_zip = spark.read.parquet("./df.parquet.gzip")
    df_brotli = spark.read.parquet("./df.parquet.brotli")
    df_none = spark.read.parquet("./df.parquet.none")
    df_zip = df_zip.sample(True, 0.1)
    df_brotli = df_brotli.sample(True, 0.1)
    df_none = df_none.sample(True, 0.1)
    j = [df_zip, df_brotli, df_none]
    
    #stores best model for each type of compression
    comp_models = []
    
    #loop for each compression model
    for df in j:
        df = df.sample(True, 0.1)
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
    
        RegParam = 0.001
        Alpha = 0.001
        Rank = 5
        #RegParam = [0.001, 0.01, 1, 10] 
        #Alpha = [0.001, 0.01, 0.1, 1]
        #Rank = [5, 10, 50 ,100]

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
                    scoreAndLabels = predictions.select('recommendations.trackNew','truth').rdd.map(tuple).repartition(1000)
                    print("scoring...")
                    metrics = RankingMetrics(scoreAndLabels)
                    print("precision...")
                    precision = metrics.precisionAt(500)
                        PRECISIONS[precision] = alsmodel
                    count += 1
                    print(f"finished {count} of {total}")
                    print(precision)
        
        

        best_precision = max(list(PRECISIONS.keys()))
        bestmodel = PRECISIONS[best_precision]
        comp_models.append(bestmodel)
    comp_models[0].write().overwrite().save(model_file1)
    comp_models[1].write().overwrite().save(model_file2)
    comp_models[2].write().overwrite().save(model_file3)
    
    print("For gzip...")    
    print(f"Best precision: {best_precision}, with regParam: {comp_models[0].getRegParam()}, alpha: {comp_models[0].getAlpha()}, rank: {comp_models[0].getRank()}")
    
    print("For brotli...")    
    print(f"Best precision: {best_precision}, with regParam: {comp_models[1].getRegParam()}, alpha: {comp_models[1].getAlpha()}, rank: {comp_models[1].getRank()}")
    
    print("For no compression...")    
    print(f"Best precision: {best_precision}, with regParam: {comp_models[2].getRegParam()}, alpha: {comp_models[2].getAlpha()}, rank: {comp_models[2].getRank()}")
    
    print("alternative models have been generated... hurrah")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('cl_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    #validation file

    val_file = sys.argv[2]

    # And the location to store the trained models
    model_file1 = sys.argv[3]
    model_file2 = sys.argv[4]
    model_file3 = sys.argv[5]



    # Call our main routine
    main(spark, data_file, val_file, model_file1, model_file2, model_file3)
