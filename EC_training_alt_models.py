'''Usage:

    $ spark-submit training.py training_data_path validation_data_path test_file_path model_file_path 

'''
# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, log

def main(spark, root, data_file, val_file, test_file, model_file1, model_file2, model_file3, model_file4, tuning = False):
    # Load the dataframe
    df = spark.read.parquet(root+data_file)
    
    df.createOrReplaceTempView("df")
    val_df = spark.read.parquet(root+val_file)
    val_df.createOrReplaceTempView("val_df")

    #we load in the test file just to be able to grab test_users
    test = spark.read.parquet(root+test_file)
    test.createOrReplaceTempView("test")
    #grab only the users present in validation and test sample because whole training dataset is too large
    df = spark.sql("SELECT * FROM df WHERE user_id IN ((SELECT user_id FROM val_df) UNION (SELECT user_id FROM test))")
    #create and store indexer info
    print("loaded data")

    #load indexers
    try:
        #to save time, indexers have been saved. If not yet created, do indexers.
        print("attempt to load indexers from file")
        indexers = PipelineModel.load("./final/indexers_all")
    except:
        user_indexer  = StringIndexer(inputCol = "user_id", outputCol = "userNew", handleInvalid = "skip")
        track_indexer = StringIndexer(inputCol = "track_id", outputCol = "trackNew", handleInvalid = "skip")
        pipeline = Pipeline(stages = [user_indexer, track_indexer]) 
        indexers = pipeline.fit(alt_df)
        indexers.write().overwrite().save("./final/indexers_all")
        print("saved indexers")
    #transform validation dataset
    val_df = indexers.transform(val_df).select(["userNew","trackNew"]).repartition(1000,"userNew")
    val_users = val_df.select("userNew").distinct().alias("userCol")
    print("transformed val")
    groundTruth = val_df.groupby("userNew").agg(F.collect_list("trackNew").alias("truth")).cache()
    print("created ground truth df")

    #transform to alternate datasets
    df_orig = df.withColumnRenamed("count","modcounts")
    #drop counts below 1
    temp = df.filter("count>1")
    df_drop1 = temp.withColumnRenamed("count","modcounts")
    #drop counts below 2
    temp = df.filter("count>2")
    df_drop2 = temp.withColumnRenamed("count","modcounts")
    #log transform counts
    df_log = df.withColumn("modcounts",log("count"))
    
    data_frames = [df_orig, df_log, df_drop1, df_drop2]

    PRECISIONS = {}
    models = []
    count = 0

    #hyperparameter tuning takes very long, so default just produces a model, set 'tuning' to True to tune
    if tuning:
        RegParam = [0.01, 0.1, 1, 10]
        Alpha = [0.1, 1, 10, 100]
        Rank = [100]
    else:
        RegParam = 10
        Alpha = 100
        Rank = 100
    
    for alt_df in data_frames:
        #transform
        alt_df = indexers.transform(alt_df).cache()
        print("transformed df")
        
        als = ALS(maxIter=10, regParam = RegParam, alpha = Alpha, rank = Rank, \
                          userCol="userNew", itemCol="trackNew", ratingCol="modcounts",\
                          coldStartStrategy="drop",implicitPrefs=True)
        alsmodel = als.fit(alt_df)

        #rec = alsmodel.recommendForUserSubset(val_users,500)
        rec = alsmodel.recommendForAllUsers(500)

        predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew, 'inner')
                
        scoreAndLabels = predictions.select('recommendations.trackNew','truth').rdd.map(tuple).repartition(1000)

        metrics = RankingMetrics(scoreAndLabels)

        precision = metrics.precisionAt(500)
        map_calc = metrics.meanAveragePrecision

        PRECISIONS[map_calc] = [precision,alsmodel,als]
        count += 1
        #print(f"finished {count} of {total}")

        print(f"precision at: {precision}, MAP: {map_calc}")
        best_map = max(list(PRECISIONS.keys()))
        best_precision,bestmodel,bestALS = PRECISIONS[best_precision]
        model.append(bestmodel)
        
    models[0].write().overwrite().save(model_file1)
    models[1].write().overwrite().save(model_file2)
    models[2].write().overwrite().save(model_file3)
    models[3].write().overwrite().save(model_file4)

    #bestALS.save("./final/alsFile")
    print(f"best MAP: {best_map}, with precision: {best_precision}, regParam: {bestALS.getRegParam}, alpha: {bestALS.getAlpha}, rank: {bestALS.getRank}")
    
    groundTruth = test.groupby("userNew").agg(F.collect_list("trackNew").alias("truth"))
    
    print("evaluating against test set")
    for i in models:
        rec = i.recommendForAllUsers(500)
        predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew, 'inner')
        scoreAndLabels = predictions.select('recommendations.trackNew','truth').rdd.map(tuple)
        metrics = RankingMetrics(scoreAndLabels)
        precision = metrics.precisionAt(500)
        map_out = metrics.meanAveragePrecision
        print(f"precision: {precision}, MAP: {map_out}")
        

    
    




# Only enter this block if we're in main
if __name__ == "__main__":
    print("starting main")
    # Create the spark session object
    spark = SparkSession.builder.appName('cfTraining').getOrCreate()
    print("created spark")
    #get root
    root = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    #validation file

    val_file = sys.argv[3]

    #test file

    test_file = sys.argv[4]

    # And the location to store the trained model
    model_file1 = sys.argv[5]
    model_file2 = sys.argv[6]
    model_file3 = sys.argv[7]
    model_file4 = sys.argv[8]
    
    # try:
    #     tuning = sys.argv[5]
    # except:
    #     tuning = True


    print("calling training function")
    # Call our main routine
    main(spark, root, data_file, val_file, test_file, model_file1, model_file2, model_file3, model_file4, tuning=False)
