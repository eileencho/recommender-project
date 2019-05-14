'''Usage:

    $ spark-submit test_file index_file model_file

'''
# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F

def main(spark, test_file, index_file, model_file):
    # Load the dataframe
    test = spark.read.parquet(test_file)
    indexer = PipelineModel.load(index_file)
    #transform user and track ids
    test = indexer.transform(test)
    #select distinct users for recommendations
    testUsers = test.select("userNew").distinct().alias("userCol")
    #establish "ground truth"
    groundTruth = test.groupby("userNew").agg(F.collect_list("trackNew").alias("truth"))
    print("created ground truth df")
    alsmodel = ALSModel.load(model_file)
    rec = alsmodel.recommendForUserSubset(testUsers,500)
    print("created recs")
    predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew, 'left')
                
    scoreAndLabels = predictions.select('recommendations.trackNew','truth').rdd.map(tuple)
    metrics = RankingMetrics(scoreAndLabels)
    precision = metrics.precisionAt(500)
    print(f"precision at 500: {precision}")

    
    




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get the filename from the command line
    test_file = sys.argv[1]

    #indexer file

    index_file = sys.argv[2]

    # And the location of trained model
    model_file = sys.argv[3]



    # Call our main routine
    main(spark, test_file, index_file, model_file)
