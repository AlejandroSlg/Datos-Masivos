import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
 
// Upload data file
val inputData = spark.read.format("libsvm").load("data/sample_multiclass_classification_data.txt")
 
// Generate the train / test division
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
 
// Instance the base classifier
val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)
 
// Instantiate the One Vs Rest classifier
val ovr = new OneVsRest().setClassifier(classifier)
 
// Train the multiclass model
val ovrModel = ovr.fit(train)
 
// Rate the model in the test data
val predictions = ovrModel.transform(test)
 
// Get evaluator
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
 
//Calculates the classification error in the test data
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
