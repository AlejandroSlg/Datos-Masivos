//Importamos las librerías
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Creamos la sesión de spark
val spark = SparkSession.builder.master("local[*]").getOrCreate()
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("sepal_length", "sepal_width", "petal_length", "petal_width","class")
val newcol = when($"class".contains("Iris-setosa"), 1.0).otherwise(when($"class".contains("Iris-virginica"), 3.0).otherwise(2.0))
val newdf = df.withColumn("etiqueta", newcol)
newdf.select("etiqueta","sepal_length", "sepal_width", "petal_length", "petal_width","class").show(150, false)

//Juntamos los datos
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width","etiqueta")).setOutputCol("features")

//Transformamos los datos
val features = assembler.transform(newdf)
features.show(150)

/* Indexar los labels que estén en el dataset para incluir todos los labels en el índice */
val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("indexedLabel").fit(features)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

/*Añadimos maxCategories para que las features cont > 4 distintos valores  sean tratadoscomo continuo */
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

//Variables de entrenamiento
val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val testData = splits(1)

/*Capa de entrada con tamaño 4 (features), dos intermediarios tamaño 5 y una de 4, de salida tamaño 3*/
val layers = Array[Int](5, 5, 4, 3)

// Crea el entrenador y establecemos los parámetros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(System.currentTimeMillis).setMaxIter(200)

//  Convierte los labels indexados devuelta a los labels originales
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

/*Encadena los indexados y la  MultilayerPerceptronClassifier en una  Pipeline. Pipeline es una técnica para implementar
el paralelismo a nivel de instrucciones dentro de un solo procesador*/
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

//Entrena el modelo
val model = pipeline.fit(trainingData)

//Predicciones
val predictions = model.transform(testData)
predictions.show(150)

// Selecciona (predicción, original label) y hace el test de error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
