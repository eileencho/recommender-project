'''Usage:

    $ spark-submit test_file index_file model_file

'''
# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql import Row
from annoy import AnnoyIndex
from time import time

def main(spark, sc, test_file, index_file, model_file, limit = 1000):
    
    # Load the dataframe
    test = spark.read.parquet(test_file)
    indexer = PipelineModel.load(index_file)
    #transform user and track ids
    test = indexer.transform(test)
    #select distinct users for recommendations, limit if needed
    testUsers = test.select("userNew").distinct().alias("userCol").limit(limit)
    #establish "ground truth"
    groundTruth = test.groupby("userNew").agg(F.collect_list("trackNew").alias("truth"))
    print("created ground truth df")
    alsmodel = ALSModel.load(model_file)

    #default version
    baseline(alsmodel, groundTruth, testUsers)
    annoy(alsmodel,groundTruth,testUsers,sc)

    trees = [20,30,40,50]
    ks = [10,50,100]

    for t in trees:
        for k in ks:
            annoy(alsmodel,groundTruth,testUsers,sc,n_trees=t,search_k=k)


def baseline(alsmodel, groundTruth, testUsers):
    print("baseline version")
    start_time = time()
    rec = alsmodel.recommendForUserSubset(testUsers,500)
    print("created recs")
    predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew, 'inner')
                
    scoreAndLabels = predictions.select('recommendations.trackNew','truth').rdd.map(tuple)
    metrics = RankingMetrics(scoreAndLabels)
    precision = metrics.precisionAt(500)
    MAP = metrics.meanAveragePrecision
    print(f"time elapsed: {time()-start_time}")
    print(f"precision at 500: {precision}")
    print(f"MAP: {MAP}")

def annoy(alsmodel, groundTruth, testUsers, sc,n_trees=10, search_k=-1):
    print(f"annoy index version with n_trees: {n_trees}, search_k: {search_k}")
    sc = SparkContext.getOrCreate()
    userfactors = alsmodel.userFactors
    size = userfactors.limit(1).select(F.size("features").alias("calc_size")).collect()[0].calc_size
    start_time = time()
    a = AnnoyIndex(size)
    for row in userfactors.collect():
        a.add_item(row.id, row.features)
    a.build(n_trees)
    a.save("./anns/annoy_t"+str(n_trees)+"_k_"+str(search_k)+".ann")
    rec_list = [(u.userNew,a.get_nns_by_item(int(u.userNew),500)) for u in testUsers.collect()]
    temp = sc.parallelize(rec_list)
    print("created recs")
    rec = spark.createDataFrame(temp,["userNew","recs"])
    predictions = rec.join(groundTruth, rec.userNew==groundTruth.userNew,'inner')

    scoreAndLabels = predictions.select('recs', 'truth').rdd.map(tuple)
    metrics = RankingMetrics(scoreAndLabels)
    precision = metrics.precisionAt(500)
    MAP = metrics.meanAveragePrecision
    print(f"time elapsed: {time()-start_time}")
    print(f"precision at 500: {precision}")
    print(f"MAP: {MAP}")
    a.unload()

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('ext2').getOrCreate()
    sc = SparkContext.getOrCreate()

    # Get the filename from the command line
    test_file = sys.argv[1]

    #indexer file

    index_file = sys.argv[2]

    # And the location of trained model
    model_file = sys.argv[3]



    # Call our main routine
    main(spark, sc, test_file, index_file, model_file)
