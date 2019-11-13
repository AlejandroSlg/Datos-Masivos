import org.apache.spark.ml.classification.LinearSVC
 
// Load training data
val training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

// LinearSVC algorithm is evaluated
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
 
// Fit model
val lsvcModel = lsvc.fit(training)
 
// Print the coefficients and intercept for linear SVC
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
