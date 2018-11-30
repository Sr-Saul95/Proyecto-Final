//Importa una sesion spar(SparkSession)
import org.apache.spark.sql.SparkSession
//Utilice el siguiente codigo para configurar errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//Crea una instancia de la sesion Spark
val spark = SparkSession.builder().getOrCreate()
//Importar Vector Assembler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
//Importa la libreria de k means para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
//Carga el DataSet de Wholesale Customers Data
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF(
  "SepalLength",
  "SepalWidth",
  "PetalLength",
  "PetalWidth",
  "class"
)
val newcol = when($"class".contains("Iris-setosa"), 1.0).
  otherwise(when($"class".contains("Iris-virginica"), 3.0).
  otherwise(2.0))
  val newdf = df.withColumn("ID", newcol)

  newdf.select("ID",
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "class").show(150, false)
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Limpieza de los datos
  //Juntando el data
  val assembler = new VectorAssembler()  .setInputCols(Array("SepalLength","SepalWidth","PetalLength","PetalWidth","ID")).setOutputCol("features")
  //Transformar datos
  val features = assembler.transform(newdf)
  features.show(5)
  // Trains a k-means model.
  val kmeans = new KMeans().setK(3).setSeed(1L).setMaxIter(100)
  val model = kmeans.fit(features)
  
  val WSSEw = model.computeCost(features)
  println(s"Within set sum of Squared Errors = $WSSEw")
  // Make predictions
  val predictions = model.transform(features)

  // Evaluate clustering by computing Silhouette score
  val evaluator = new ClusteringEvaluator()

  val silhouette = evaluator.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // Shows the result.
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)
