package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object BostonCrimesMap extends App {
  if (args.length != 3) {
    println("args are not correct")
    sys.exit(1)
  }
  val pathToCrimes = args(0)
  val pathToOffenseCodes = args(1)
  val pathToOutput = args(2)

  val spark = SparkSession.builder()
    .appName("Boston Crimes")
    .master("local[*]")
    .getOrCreate()
  val sc = spark.sparkContext

  import spark.implicits._

  val crimes = spark.read.option("header", "true").csv(pathToCrimes)

  val crimesTotal = crimes.groupBy("DISTRICT").count()

  crimes.createOrReplaceTempView("crimes")
  val crimesMonthly = spark.sql(
    "SELECT DISTRICT, percentile_approx(incidents, 0.5) AS median FROM " +
      " (SELECT DISTRICT, concat(YEAR, MONTH) AS year_month, COUNT(*) AS incidents" +
      " FROM crimes" +
      " GROUP BY DISTRICT, year_month) t GROUP BY DISTRICT")

  val offenseCodes = spark.read.option("header", "true").csv(pathToOffenseCodes)
    .groupBy("CODE").agg(first(trim(split(col("NAME"), "-").getItem(0))).as("CRIME_TYPE"))
  val frequentCrimeTypes = crimes.join(broadcast(offenseCodes), crimes("OFFENSE_CODE") <=> offenseCodes("CODE"))
    .groupBy($"DISTRICT", $"CRIME_TYPE")
    .agg(count("CRIME_TYPE").as("frequency")).orderBy($"DISTRICT".asc, $"frequency".desc)
    .groupBy($"DISTRICT")
    .agg(concat_ws(", ", slice(collect_list($"CRIME_TYPE"), 1, 3)).as("TOP_CRIME_TYPES"))

  val lat = crimes.groupBy($"DISTRICT").agg(avg("LAT").as("lat"))
  val lng = crimes.groupBy($"DISTRICT").agg(avg("LONG").as("lng"))

  val crimesAggregate = crimesTotal
    .join(crimesMonthly, "DISTRICT")
    .join(frequentCrimeTypes, "DISTRICT")
    .join(lat, "DISTRICT")
    .join(lng, "DISTRICT")
    .select(
      crimesTotal("DISTRICT"),
      crimesTotal("count").as("crimes_total"),
      crimesMonthly("median").as("crimes_monthly"),
      frequentCrimeTypes("TOP_CRIME_TYPES").as("frequent_crime_types"),
      lat("lat"),
      lng("lng")
    )

  crimesAggregate.toDF().repartition(1).write.parquet(pathToOutput)

  spark.stop()
}