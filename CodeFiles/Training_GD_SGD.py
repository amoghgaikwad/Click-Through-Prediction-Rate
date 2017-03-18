from pyspark.sql import Row
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import BinaryClassificationMetrics


#sc = spark.sparkContext
conf = SparkConf().setAppName("Assignment 3")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)


###### Part 1- Preparing the data #######

# To count the number of distinct values in each column
distnct_values = {}

# Load a text file and convert each line to a Row.
lines = sc.textFile("train")

# Extract header
header = lines.first()
lines = lines.filter(lambda row : row != header)

parts = lines.map(lambda l: l.split(","))
data_T = parts.map(lambda p: Row(click=(p[1]), C1=p[3], banner_pos = p[4], site_id=p[5], site_domain=p[6], 
	site_category=p[7], app_id=p[8], app_domain=p[9], app_category=p[10], device_id=p[11], device_ip=p[12], device_model=p[13], device_type=p[14],
	device_conn_type=p[15], C14=p[16], C15=p[17], C16=p[18], C17=p[19], C18=p[20], C19=p[21], C20=p[22], C21=p[23] ))

# Create the dataframe
df = sqlCtx.createDataFrame(data_T)

# Selecting all the categorical columns and checking the distinct values in each
col_names= ['C1','site_category','app_category','device_type','C14','C15','C16','C17','C18','C19','C20','C21']
for i in col_names: ##col_names contains names of cols that contains categorical data
    distinctValues = df.select(i).distinct().rdd.map(lambda r: r[0]).count()
    distnct_values[i] = distinctValues

# Delete columns which have more than 100 distinct values in them
for key, value in distnct_values.iteritems() :
     if int(value) >100:
             df = df.drop(str(key))

# Drop the columns which have NA values in it.
df = df.na.replace('', 'NA', 'C1')
df = df.dropna()

# StringIndexer on all the categorical columns
c1I = StringIndexer(inputCol="C1", outputCol="iC1", handleInvalid="skip")
c15I = StringIndexer(inputCol="C15", outputCol="iC15", handleInvalid="skip")
c16I = StringIndexer(inputCol="C16", outputCol="iC16", handleInvalid="skip")
c18I = StringIndexer(inputCol="C18", outputCol="iC18", handleInvalid="skip")
c19I = StringIndexer(inputCol="C19", outputCol="iC19", handleInvalid="skip")
c21I = StringIndexer(inputCol="C21", outputCol="iC21", handleInvalid="skip")
appcatI = StringIndexer(inputCol="app_category", outputCol="i_app_category", handleInvalid="skip")
devtypeI = StringIndexer(inputCol="device_type", outputCol="i_device_type", handleInvalid="skip")
sitecatI = StringIndexer(inputCol="site_category", outputCol="i_site_category", handleInvalid="skip")


# OneHotEncoder applied after the stringIndexer to form binary vector for each column
c1E = OneHotEncoder(inputCol="iC1", outputCol="C1Vector")
c15E = OneHotEncoder(inputCol="iC15", outputCol="C15Vector")
c16E = OneHotEncoder(inputCol="iC16", outputCol="C16Vector")
c18E = OneHotEncoder(inputCol="iC18", outputCol="C18Vector")
c19E = OneHotEncoder(inputCol="iC19", outputCol="C19Vector")
c21E = OneHotEncoder(inputCol="iC21", outputCol="C21Vector")
appcatE = OneHotEncoder(inputCol="i_app_category", outputCol="i_app_category_Vector")
devtypeE = OneHotEncoder(inputCol="i_device_type", outputCol="i_device_type_Vector")
sitecatE = OneHotEncoder(inputCol="i_site_category", outputCol="i_site_category_Vector")

# Vector assembler
Assembler = VectorAssembler(
    inputCols=["C1Vector", "C15Vector", "C16Vector", "C18Vector", "C19Vector", "C21Vector", "i_app_category_Vector", "i_device_type_Vector", "i_site_category_Vector"],
    outputCol="features")

# Pipeline to sum up all the stringIndexers and OneHotEncoders and VectorAssemebler
data_P = Pipeline(stages=[c1I, c15I, c16I, c18I, c19I, c21I, appcatI, devtypeI, sitecatI, 
	c1E, c15E, c16E, c18E, c19E, c21E, appcatE, devtypeE, sitecatE, Assembler])

data_m = data_P.fit(df)
data_t = data_m.transform(df)

###### Part 1 ends here #####

# Making the labelpoints to train the data with LR
parsedData=data_t.select('click', 'features').rdd.map(lambda row: LabeledPoint(float(row.click),Vectors.dense((row.features).toArray())))

# Split the dataset
training,test = parsedData.randomSplit([0.6, 0.4], seed=11L)
training.cache()

# Train the data using a version of logistic regression that optimizes the parameters with Gradient Descent(GD)
# This is done by setting the parameters -step=1, miniBatchFraction=1, while training the dataset.
model = LogisticRegressionWithSGD.train(training, step=1, miniBatchFraction=1, regType=None)

# To Train with SGD : The following model will be used
# Train the data using a version of logistic regression that optimizes the parameters with Stochastic Gradient Descent(SGD)

#----> # model = LogisticRegressionWithSGD.train(training, step=0.1, miniBatchFraction=0.1, regType=None)
