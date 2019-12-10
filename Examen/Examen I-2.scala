//1
import org.apache.spark.sql.SparkSession
var spark = SparkSession.builder().getOrCreate()

//2
var df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//3
df.columns.sorted()

//4
df.printSchema()

//5
df.select($"Date", $"Open", $"High", $"Low", $"Close").show()
df.head(5)


//6
df.describe().show()

//7
var dfp7 = df.withColumn("HV Ratio",df("High")/df("Volume"))
dfp7.show()

//9
//Es el valor de las acciones de Netflix con el cual termino el dìa entre los años 2011- 2016

//10
df.select(max ($"Volume")).show()
df.select(min ($"Volume")).show()

//11

//a
var p11a = df.filter($"Close" < 600).count()

//b
var p11b = (df.filter($"High" > 500).count()*100)/1260

//c
df.select(corr("High", "Volume")).show()

//d
var dfyear=df.withColumn("Year",year(df("Date")))
var dfmax=dfyear.groupBy("Year").max()
dfmax.select($"Year",$"max(High)").show()

//e
df.groupBy(month(df("Date"))).avg("Close").orderBy(month(df("Date"))).show()