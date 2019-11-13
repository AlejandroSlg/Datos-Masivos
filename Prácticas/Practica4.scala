import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
 
// Load and analyze the data file, converting it into a DataFrame
val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
 
// Index of labels, adding metadata to the column of labels.
// It fits the entire data set to include all labels in the index
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical characteristics and indicate them.
// Set maxCategories so that entities with> 4 different values are treated as continuous
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
 
// Divide the data into training and test sets (30% for tests)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
 
// Train a GBT model
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
 
// Convert indexed tags back to original tags
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
 
// Chain and GBT indexers in a pipe
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
 
// Train model.
val model = pipeline.fit(trainingData)
 
// Predictions are made
val predictions = model.transform(testData)
 
// Select example rows to display
predictions.select("predictedLabel", "label", "features").show(5)
 
// Select (prediction, true label) and calculate the test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")
 
// GBT model is generated
val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
