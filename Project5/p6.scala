val df = spark.read.json("tweets_#superbowl.txt")
df.printSchema()
df.createOrReplaceTempView("table")
val d = spark.sql("SELECT title,tweet.user.location FROM table" )
d.repartition(1).write.json("title_location")
// rename partitionXXX.json file to title_location.json

val df = spark.read.json("title_location/title_location.json")
df.printSchema()
df.createOrReplaceTempView("table")
val d = spark.sql("SELECT title, location FROM table WHERE location REGEXP '.*MA.*'OR location REGEXP '.*Mass.*' OR location REGEXP '.*WA.*' OR location REGEXP '.*Wash.*'")
d.repartition(1).write.json("title_location_1")
// rename partitionXXX.json file to title_location_1.json

val df = spark.read.json("title_location_1/title_location_1.json")
df.printSchema()
df.createOrReplaceTempView("table")
val d = spark.sql("SELECT title, location FROM table WHERE location NOT REGEXP '.*DC.*' AND location NOT REGEXP '.*D\\.C\\..*'")
d.repartition(1).write.json("final_data")




