import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
 
// Load the data stored in LIBSVM format as a DataFrame
val data = spark.read.format("libsvm").load("data/sample_multiclass_classification_data.txt")
 
// Divide the data by train and test
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
 
// Specify layers for the neural network:
// Input layer size 4 (features), two intermediate sizes 5 and 4
// and output size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)
 
// Create the coach and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
 
// Train the model jeje
val model = trainer.fit(train)
 
// Calculate the accuracy in the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")

// The accuracy of the predictions is evaluated
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
 
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
