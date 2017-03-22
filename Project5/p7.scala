// run each line in spark-shell

// Topic 1 retrieve title_time dataset
val df = spark.read.json("tweets_#superbowl.txt")
df.createOrReplaceTempView("table")
val d = spark.sql("SELECT title,firstpost_date FROM table" )
d.repartition(1).write.json("title_time")


// Topic 2
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, IDF, Tokenizer}

val df = spark.read.json("final_data/p6_data.json")
df.printSchema()
val labeleddf = df.withColumn("label", df("location").rlike(".*MA.*|.*Mass.*").cast("double"))
labeleddf.sample(false,0.003,10L).show(10)
val Array(training, test) = labeleddf.randomSplit(Array(0.9, 0.1), seed = 12345)
println("Total Document Count = " + labeleddf.count())
println("Training Count = " + training.count() + ", " + training.count*100/(labeleddf.count()).toDouble + "%")
println("Test Count = " + test.count() + ", " + test.count*100/(labeleddf.count().toDouble) + "%")

val tokenizer = new Tokenizer().setInputCol("title").setOutputCol("words")
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(false)
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
val lr = new LogisticRegression().setRegParam(0.01).setThreshold(0.5)
val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, lr))

val model = pipeline.fit(training)
val predictions = model.transform(test)
predictions.select("location", "probability", "prediction", "label").sample(false,0.01,10L).show(10)

val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
println("Area under the ROC curve = " + evaluator.evaluate(predictions))