/*Importamos las librerías*/
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

/*Se importan los datos del CSV*/
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

/*Se categoriza las variables de string a valor numérico*/
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val clean = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cleanData = clean.withColumn("y",'y.cast("Int"))

/*Creación del Array con los datos seleccionados*/
val featureCols = Array("age","previous","balance","duration")

/*Creación del Vector en base a los features*/
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

/*Transformación a un nuevo DF*/
val df2 = assembler.transform(cleanData)

/*Rename de columnas*/
val featuresLabel = df2.withColumnRenamed("y", "label")

/*Selección de index*/
val dataI = featuresLabel.select("label","features")

/*Creación del Array con los datos de entrenamiento y test*/
val Array(training, test) = dataI.randomSplit(Array(0.7, 0.3), seed = 12345)

/*Modelo de Regresion*/
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

val lrModel = lr.fit(training)

/*Impresión de los coeficientes e intercepciones*/
println(s"Coefficients: \n${lrModel.coefficientMatrix}")
println(s"Intercepts: \n${lrModel.interceptVector}")


/*Impresión de la precisión*/
println(s"Accuracy: $accuracy")
