# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px; width:30%; height: auto;" src="https://i.imgur.com/pQvJTVf.jpeg"   />   
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          07-01-Machine Learning at Scale with the NYC Taxi Dataset
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-28-2024
# MAGIC
# MAGIC    Purpose:   This notebook demonstrates machine learning at scale with Spark and MLlib
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Data Ingestion and Initial Exploration
# MAGIC            1-1 - Load the data into a Spark DataFrame
# MAGIC            1-2 - Get an estimate on the row count by sampling
# MAGIC            1-3 - Perform Type Conversions
# MAGIC         2 - Data Preprocessing at Scale
# MAGIC            2-1 - Add a trip_duration column 
# MAGIC            2-2 - Filter out trips with unrealistic durations or distances
# MAGIC            2-3 - Drop the rows with missing critical values
# MAGIC         3 - Perform Feature Engineering
# MAGIC            3-1 - Extract the Temporal Features
# MAGIC            3-2 - Impute the mean in the numerical features
# MAGIC            3-3 - Handle Missing Values in Categorical Features
# MAGIC         4 - Feature Transformation and Vectorization
# MAGIC            4-1 - Setting up Indexers and Encoders
# MAGIC            4-2 - Setting up the VectorAssembler
# MAGIC            4-3 - Executing the Indexers
# MAGIC            4-4 - Executing the Encoders
# MAGIC            4-5 - Using the VectorAssember to create our Features Column
# MAGIC            4-6 - Finalizing the Dataset
# MAGIC         5 - Model Training at Scale
# MAGIC            5-1 - Splitting the Data
# MAGIC            5-2 - Training a Gradient Boosted Tree Regressor
# MAGIC            5-3 - Evaluating the Model
# MAGIC               
# MAGIC

# COMMAND ----------

# MAGIC %run "../common/Constants"

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col, isnan, when, count, date_format, hour, dayofweek, unix_timestamp

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer

# COMMAND ----------

# MAGIC %md
# MAGIC #Data Ingestion and Initial Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load the data into a Spark DataFrame

# COMMAND ----------

# Define the list of Parquet file paths for each month of 2023
# The f-string formats the file paths dynamically from 'nyc_taxi_data' directory
# The 'i:02' formats the month part to be two digits (e.g., '01', '02', ..., '12')
files = [f"dbfs:{NYC_TAXI_DATASET_PATH}/yellow_tripdata_2023_{i:02}.parquet" for i in range(1, 13)]

# Initialize an empty list to hold DataFrames for each file
dfs = []

# Loop through each file in the list
for file in files:
    # Read the current Parquet file into a Spark DataFrame
    df = spark.read.parquet(file)

    # Cast the 'passenger_count' column to 'double' to ensure consistent data type across all files
    df = df.withColumn("passenger_count", df["passenger_count"].cast("double"))

    # Append the adjusted DataFrame to the list
    dfs.append(df)

# Initialize the final DataFrame with the first DataFrame in the list
taxi_df = dfs[0]

# Loop through the remaining DataFrames and perform a union operation
for df in dfs[1:]:
    taxi_df = taxi_df.union(df)

# Display the schema of the combined DataFrame to verify the data structure
taxi_df.printSchema()

# Display the first 5 rows of the combined DataFrame for a quick data check
display(taxi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Get an estimate on the row count by sampling

# COMMAND ----------

# Define a sample fraction to estimate the number of rows in the DataFrame
# A fraction of 0.1 means a 10% sample of the total dataset will be used
# Adjust the fraction as needed for more or less sampling accuracy
sample_fraction = 0.1  

# Calculate the approximate row count by:
# 1. Taking a sample of the DataFrame using the specified fraction
# 2. Counting the number of rows in the sample
# 3. Dividing the sample count by the fraction to estimate the total number of rows
approx_row_count = taxi_df.sample(fraction=sample_fraction).count() / sample_fraction

# Print the approximate number of rows, formatted with commas for better readability
print(f"Approximate number of rows: {approx_row_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Perform Type Conversions

# COMMAND ----------

# MAGIC %md
# MAGIC From the above, we can see we don't have correct datatypes for all columns, so let's convert the the correct types. We need to do the following conversions:
# MAGIC
# MAGIC * **Timestamps**: Convert to timestamp type.
# MAGIC * **IDs**: Convert to integer type.
# MAGIC * **Categorical columns** (e.g., store_and_fwd_flag): Leave as string.
# MAGIC * **Payment-related columns**: Convert to float for monetary amounts.

# COMMAND ----------

from pyspark.sql.functions import col

# Convert columns to the correct data types
taxi_df = taxi_df \
    .withColumn("VendorID", col("VendorID").cast("integer")) \
    .withColumn("tpep_pickup_datetime", col("tpep_pickup_datetime").cast("timestamp")) \
    .withColumn("tpep_dropoff_datetime", col("tpep_dropoff_datetime").cast("timestamp")) \
    .withColumn("passenger_count", col("passenger_count").cast("integer")) \
    .withColumn("trip_distance", col("trip_distance").cast("float")) \
    .withColumn("RatecodeID", col("RatecodeID").cast("integer")) \
    .withColumn("PULocationID", col("PULocationID").cast("integer")) \
    .withColumn("DOLocationID", col("DOLocationID").cast("integer")) \
    .withColumn("payment_type", col("payment_type").cast("integer")) \
    .withColumn("fare_amount", col("fare_amount").cast("float")) \
    .withColumn("extra", col("extra").cast("float")) \
    .withColumn("mta_tax", col("mta_tax").cast("float")) \
    .withColumn("tip_amount", col("tip_amount").cast("float")) \
    .withColumn("tolls_amount", col("tolls_amount").cast("float")) \
    .withColumn("improvement_surcharge", col("improvement_surcharge").cast("float")) \
    .withColumn("total_amount", col("total_amount").cast("float")) \
    .withColumn("congestion_surcharge", col("congestion_surcharge").cast("float")) \
    .withColumn("airport_fee", col("airport_fee").cast("float"))

# Show the updated schema
taxi_df.printSchema()

# Show a few rows to verify the changes
display(taxi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Data Preprocessing at Scale

# COMMAND ----------

# MAGIC %md
# MAGIC ##Add a trip_duration column 

# COMMAND ----------

# Calculate trip duration in minutes
# The 'unix_timestamp()' function converts the timestamp columns to Unix time (seconds since 1970-01-01 00:00:00 UTC)
# Subtracting 'tpep_pickup_datetime' from 'tpep_dropoff_datetime' gives the trip duration in seconds
# Dividing the difference by 60 converts the duration from seconds to minutes
taxi_df = taxi_df.withColumn(
    'trip_duration',
    (unix_timestamp('tpep_dropoff_datetime') - unix_timestamp('tpep_pickup_datetime')) / 60
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Filter out trips with unrealistic durations or distances

# COMMAND ----------

from pyspark.sql.functions import col

# Filter out trips with unrealistic durations
# Only keep rows where 'trip_duration' is greater than 0 and less than or equal to 240 minutes (4 hours)
# This removes trips with negative, zero, or excessively long durations
taxi_df = taxi_df.filter((col('trip_duration') > 0) & (col('trip_duration') <= 240))

# Filter out trips with unrealistic distances
# Only keep rows where 'trip_distance' is greater than 0 and less than or equal to 100 miles
# This removes trips with negative, zero, or unusually long distances
taxi_df = taxi_df.filter((col('trip_distance') > 0) & (col('trip_distance') <= 100))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Drop the rows with missing critical values

# COMMAND ----------

# Handle missing values by removing rows with nulls in critical columns

# Use 'dropna()' to drop rows where any of the specified columns have missing (null) values
# The 'subset' argument specifies the list of columns to check for missing values
# The columns listed are considered critical for analysis:
# 1. 'tpep_pickup_datetime' and 'tpep_dropoff_datetime': Essential for calculating trip duration
# 2. 'trip_distance': Represents the distance traveled, crucial for fare estimation and analysis
# 3. 'passenger_count': Important for understanding passenger volume and capacity
# 4. 'PULocationID' and 'DOLocationID': Represent pickup and drop-off locations, necessary for location-based analysis
# 5. 'trip_duration': Calculated column used for various analyses, including trip speed and efficiency

taxi_df = taxi_df.dropna(
    subset=[
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'trip_distance',
        'passenger_count',
        'PULocationID',
        'DOLocationID',
        'trip_duration'
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Perform Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ##Extract the Temporal Features

# COMMAND ----------

from pyspark.sql.functions import col, hour, dayofweek, date_format
from pyspark.sql.types import IntegerType

# Extract the hour of the pickup time
# The 'hour()' function extracts the hour from 'tpep_pickup_datetime' as an integer (0-23)
# Creates a new column 'pickup_hour' to represent the hour of day when the trip started
taxi_df = taxi_df.withColumn('pickup_hour', hour(col('tpep_pickup_datetime')))

# Extract the day of the week from the pickup time
# The 'dayofweek()' function extracts the day of the week as an integer (1 = Sunday, ..., 7 = Saturday)
# Creates a new column 'pickup_day' to represent the day of the week when the trip started
taxi_df = taxi_df.withColumn('pickup_day', dayofweek(col('tpep_pickup_datetime')))

# Extract the month from the pickup time
# The 'date_format()' function extracts the month from 'tpep_pickup_datetime' as a string
# The 'M' format specifier represents the month (1-12), and 'cast(IntegerType())' converts it to integer
# Creates a new column 'pickup_month' to represent the month when the trip started
taxi_df = taxi_df.withColumn('pickup_month', date_format(col('tpep_pickup_datetime'), 'M').cast(IntegerType()))

# Extract the year from the pickup time
# The 'date_format()' function extracts the year from 'tpep_pickup_datetime' as a string
# The 'yyyy' format specifier represents the 4-digit year, and 'cast(IntegerType())' converts it to integer
# Creates a new column 'pickup_year' to represent the year when the trip started
taxi_df = taxi_df.withColumn('pickup_year', date_format(col('tpep_pickup_datetime'), 'yyyy').cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Impute the mean in the numerical features

# COMMAND ----------

# Define the list of numerical columns to handle missing values
# These columns are critical numerical features for analysis:
# 1. 'trip_distance': Represents the distance traveled during the trip
# 2. 'passenger_count': Represents the number of passengers in the trip
# 3. 'pickup_hour': Represents the hour when the trip started
numerical_cols = ['trip_distance', 'passenger_count', 'pickup_hour']

# Create an Imputer instance for handling missing values
# 'inputCols' specifies the list of columns to impute
# 'outputCols' specifies the columns to store the imputed values (replaces the original columns)
# 'setStrategy("mean")' sets the strategy to use the mean of each column for imputation
imputer = Imputer(inputCols=numerical_cols, outputCols=numerical_cols).setStrategy("mean")

# Fit the Imputer model to the DataFrame and transform it
# This replaces missing values in the specified columns with their respective means
taxi_df = imputer.fit(taxi_df).transform(taxi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Handle Missing Values in Categorical Features

# COMMAND ----------

# Define the list of categorical columns to handle missing values
# These columns represent categorical features related to trip data:
# 1. 'PULocationID': Represents the pickup location ID
# 2. 'DOLocationID': Represents the drop-off location ID
# 3. 'pickup_day': Represents the day of the week for the pickup
categorical_columns = ['PULocationID', 'DOLocationID', 'pickup_day']

# Fill nulls in the specified categorical columns with a default value of -1
# Using '-1' as the default value indicates missing or unknown values in these columns
# The dictionary comprehension creates a mapping of each column to the default value (-1)
taxi_df = taxi_df.fillna({col: -1 for col in categorical_columns})

# COMMAND ----------

# MAGIC %md
# MAGIC #Feature Transformation and Vectorization

# COMMAND ----------

# MAGIC %md
# MAGIC ##Setting up Indexers and Encoders

# COMMAND ----------

# StringIndexer and OneHotEncoder for handling categorical variables
# Create a list of StringIndexer instances for each categorical column
# 'inputCol' specifies the original column, and 'outputCol' creates a new column with '_indexed' suffix
# 'handleInvalid='keep'' ensures that any unseen or missing categories during transformation are assigned a separate index
indexers = [
    StringIndexer(inputCol=column, outputCol=column + "_indexed", handleInvalid='keep')
    for column in categorical_columns
]

# Create a list of OneHotEncoder instances for each indexed categorical column
# 'inputCol' is the indexed column from StringIndexer, and 'outputCol' creates a new column with '_encoded' suffix
# 'handleInvalid='keep'' ensures that any invalid values (e.g., unseen categories) are handled by creating a new category
encoders = [
    OneHotEncoder(inputCol=column + "_indexed", outputCol=column + "_encoded", handleInvalid='keep')
    for column in categorical_columns
]

# COMMAND ----------

# Print out the number of indexers and encoders
print(f"Number of Indexers: {len(indexers)}")
print(f"Number of Encoders: {len(encoders)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Setting up the VectorAssembler

# COMMAND ----------


# Define the input columns for the assembler
assembler_input_cols = [column + "_encoded" for column in categorical_columns] + numerical_cols

# VectorAssembler for combining all numerical and encoded categorical features into a single 'features' column
# 'inputCols' specifies the list of columns to be combined into a feature vector
# Includes both numerical columns and encoded categorical columns
# 'handleInvalid='skip'' skips rows with nulls in the specified columns during vector assembly
assembler = VectorAssembler(
    inputCols=assembler_input_cols,
    outputCol='features',
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Executing the indexers

# COMMAND ----------

# Start with the original DataFrame
df_indexed = taxi_df  

# Loop over each indexer
for indexer in indexers:
    df_indexed = indexer.fit(df_indexed).transform(df_indexed)

# Print the resulting schema
df_indexed.printSchema()  # Verify indexed columns are created


# COMMAND ----------

# MAGIC %md
# MAGIC ##Executing the Encoders

# COMMAND ----------

# Start with the indexed DataFrame
df_encoded = df_indexed 

# Loop over each encoder
for encoder in encoders:
    df_encoded = encoder.fit(df_encoded).transform(df_encoded)

# Print out the resulting schema
df_encoded.printSchema()  

# COMMAND ----------

# MAGIC %md
# MAGIC ##Using the VectorAssember to create our Features Column

# COMMAND ----------

# Step 3: Apply VectorAssembler
df_assembled = assembler.transform(df_encoded)

# The schema Should now contain 'features'
df_assembled.printSchema() 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Finalizing the Dataset
# MAGIC In MLlib, the **features** column contains the features, and the target variable is named **label**

# COMMAND ----------

# Select the final dataset with only the 'features' column and the target column 'trip_duration'
# The 'features' column contains the vector of all input features (numerical + one-hot encoded categorical)
# The 'trip_duration' column serves as the target variable for the model
df_final = df_assembled.select(['features', 'trip_duration'])
# Rename the 'trip_duration' column to 'label' for compatibility with Spark ML models
# Spark ML expects the target variable to be named 'label' by default during training
df_final = df_final.withColumnRenamed('trip_duration', 'label')

# Print out the final columns
df_final.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #Efficient Storage with Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC ##Writing to Delta Lake

# COMMAND ----------

# Construct the Delta file path using the DELTA_NYC_DATASET_PATH variable
# The f-string format is used to create a valid DBFS path for saving the Delta table
delta_path = f"dbfs:{DELTA_NYC_DATASET_PATH}"

# Print the constructed Delta file path for verification
print(f"The Delta file path is: {delta_path}")

# Remove all files in the specified directory before writing new data
# 'dbutils.fs.rm()' is used to delete the directory recursively, ensuring it's empty
# 'recurse=True' allows for recursive deletion of all files and subdirectories within the path
dbutils.fs.rm(delta_path, recurse=True)

# Write the prepared data (df_prepared) to a Delta Lake table
# 'format("delta")' specifies that the data should be written in Delta format
# 'mode("overwrite")' ensures that any existing data in the directory is replaced by the new data
# 'save(delta_path)' writes the Delta table to the specified path
df_final.write.format("delta").mode("overwrite").save(delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Optimize the Delta Table

# COMMAND ----------

# Optimize the Delta table to improve performance
# The 'OPTIMIZE' command is a Delta Lake operation that reorganizes the data files within the specified Delta table
# It compacts small files into larger files, reducing the number of files needed for reads
# This operation improves query performance by reducing I/O and enabling faster data scans
# The f-string dynamically inserts the path of the Delta table to be optimized
spark.sql(f"OPTIMIZE '{DELTA_NYC_DATASET_PATH}'")

# COMMAND ----------

# MAGIC %md
# MAGIC #Model Training at Scale

# COMMAND ----------

# MAGIC %md
# MAGIC ##Splitting the Data

# COMMAND ----------

# Split the DataFrame into training and test sets using random sampling
# 'randomSplit([0.8, 0.2], seed=42)' splits the data into two parts:
# - 80% for training ('train_data')
# - 20% for testing ('test_data')
# The 'seed=42' sets a random seed to ensure the split is reproducible, 
# making it consistent across different runs for reliability in model evaluation
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Training a Gradient Boosted Tree Regressor

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize the Gradient-Boosted Tree (GBT) Regressor
# 'GBTRegressor' is a machine learning algorithm that builds an ensemble of decision trees 
# to improve prediction accuracy for regression tasks
# 'featuresCol' specifies the column containing the input features ('features')
# 'labelCol' specifies the target variable ('label'), which is 'trip_duration' in this case
# 'maxIter=50' sets the number of boosting iterations (i.e., the number of trees to be built)
gbt = GBTRegressor(featuresCol='features', labelCol='label', maxIter=50)

# Train the GBT model using the training data
# The 'fit()' method trains the model on the 'train_data' DataFrame
# This step builds the ensemble model by iteratively fitting decision trees and combining their outputs
gbt_model = gbt.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluating the Model

# COMMAND ----------

# Make predictions on the test set using the trained GBT model
# The 'transform()' method generates predictions for the 'test_data' DataFrame
# It adds a new 'prediction' column to the DataFrame, containing the model's predicted values for 'trip_duration'
predictions = gbt_model.transform(test_data)

# Initialize the RegressionEvaluator for model evaluation
# 'RegressionEvaluator' is used to assess the performance of the regression model
# 'labelCol' specifies the actual target column ('label')
# 'predictionCol' specifies the column with predicted values ('prediction')
# 'metricName' is set to 'rmse' (Root Mean Squared Error), a common metric for measuring prediction accuracy
evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')

# Evaluate the model's performance on the test set by calculating the RMSE
# The 'evaluate()' method computes the RMSE between the actual and predicted values
# RMSE measures the average difference between predicted and actual values, indicating the model's error in minutes
rmse = evaluator.evaluate(predictions)

# Print the RMSE value for the test set, formatted to two decimal places
print(f"Test RMSE: {rmse:.2f} minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC #Hyperparameter Tuning with HyperOpt
# MAGIC Optimize model performance by tuning hyperparameters using HyperOpt with SparkTrials.

# COMMAND ----------

import sys
sys.setrecursionlimit(10000)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Defining the Objective Function

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from hyperopt import STATUS_OK

def objective_function(params):
    try:
        # Convert hyperparameters to integers
        maxDepth = int(params['maxDepth'])
        maxIter = int(params['maxIter'])

        # Access the broadcasted training and test data
        train_data = train_data_bc.value
        test_data = test_data_bc.value

        # Initialize a Gradient-Boosted Tree Regressor with the current hyperparameters
        gbt = GBTRegressor(
            featuresCol='features',
            labelCol='label',
            maxDepth=maxDepth,
            maxIter=maxIter
        )

        # Train the model using the broadcasted training data
        model = gbt.fit(train_data)

        # Make predictions on the broadcasted test data
        predictions = model.transform(test_data)

        # Calculate the RMSE for model evaluation
        evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
        rmse = evaluator.evaluate(predictions)

        # Return the evaluation result with RMSE as 'loss'
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        # Return a large loss if any error occurs
        return {'loss': float('inf'), 'status': STATUS_OK}


# COMMAND ----------

# MAGIC %md
# MAGIC ##Defining the Search Space

# COMMAND ----------

from hyperopt import hp

# Define the hyperparameter search space for tuning the Gradient-Boosted Tree (GBT) model
search_space = {
    # 'maxDepth' defines the maximum depth of each decision tree in the ensemble
    # 'hp.quniform()' creates a discrete uniform distribution, meaning it will generate integer values
    # between 5 and 15 (inclusive) with a step size of 1
    # This helps explore different tree depths, controlling the model's complexity
    'maxDepth': hp.quniform('maxDepth', 5, 15, 1),
    
    # 'maxIter' defines the maximum number of boosting iterations (i.e., number of trees in the model)
    # 'hp.quniform()' creates a discrete uniform distribution between 50 and 150 with a step size of 10
    # This helps explore different numbers of trees, affecting model accuracy and training time
    'maxIter': hp.quniform('maxIter', 50, 150, 10)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ##Running the HyperOpt with SparkTrials

# COMMAND ----------

from hyperopt import fmin, tpe, SparkTrials

# Initialize SparkTrials for parallel execution
spark_trials = SparkTrials(parallelism=4)

# Perform hyperparameter tuning with Hyperopt and SparkTrials
best_hyperparams = fmin(
    fn=objective_function,           # Objective function
    space=search_space,              # Search space
    algo=tpe.suggest,                # Optimization algorithm
    max_evals=20,                    # Number of evaluations
    trials=spark_trials              # Use SparkTrials for distributed execution
)

# Print the best hyperparameters found
print("Best hyperparameters:", best_hyperparams)


# COMMAND ----------

# MAGIC %md
# MAGIC #Experiment Tracking wiht MLFlow
# MAGIC Track experiments, parameters, metrics, and models using MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Enabling MLflow AutoLogging

# COMMAND ----------

import mlflow
import mlflow.spark

# Enable automatic logging of Spark ML model training and metrics with MLflow
# 'mlflow.spark.autolog()' sets up automatic tracking of Spark ML experiments
# It logs model parameters, metrics (e.g., RMSE), and models to the MLflow tracking server
# This includes tracking the start and end times of training, hyperparameters, and evaluation metrics
mlflow.spark.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Logging with the Objective Function

# COMMAND ----------

def objective_function(params):
    # Start a new MLflow run for this trial, setting 'nested=True' to allow nesting within a parent run
    # This ensures that each hyperparameter combination is tracked as a separate MLflow run
    with mlflow.start_run(nested=True):
        # Extract hyperparameters from the 'params' dictionary and convert to integers
        # 'maxDepth' controls the maximum depth of each decision tree
        # 'maxIter' sets the number of boosting iterations (i.e., number of trees in the ensemble)
        maxDepth = int(params['maxDepth'])
        maxIter = int(params['maxIter'])
        
        # Initialize the Gradient-Boosted Tree (GBT) Regressor with the specified hyperparameters
        # 'featuresCol' specifies the input feature vector ('features')
        # 'labelCol' specifies the target variable ('label')
        gbt = GBTRegressor(
            featuresCol='features',
            labelCol='label',
            maxDepth=maxDepth,
            maxIter=maxIter
        )
        
        # Train the GBT model using the training data with the current hyperparameters
        model = gbt.fit(train_data)
        
        # Make predictions on the test data using the trained model
        predictions = model.transform(test_data)
        
        # Evaluate the model's performance by calculating Root Mean Squared Error (RMSE)
        # The 'evaluator' is a RegressionEvaluator instance set up to measure RMSE
        rmse = evaluator.evaluate(predictions)
        
        # Log hyperparameters and evaluation metrics to MLflow for tracking
        mlflow.log_param('maxDepth', maxDepth)  # Log the maximum tree depth
        mlflow.log_param('maxIter', maxIter)    # Log the number of iterations
        mlflow.log_metric('rmse', rmse)         # Log the RMSE metric
        
        # Return the evaluation result as a dictionary
        # 'loss' represents the RMSE, which is minimized during hyperparameter tuning
        # 'status' is set to STATUS_OK to indicate a successful evaluation
        return {'loss': rmse, 'status': STATUS_OK}



# COMMAND ----------

# MAGIC %md
# MAGIC #Resource Management and Cluster Configuration
# MAGIC Optimizing cluster resources ensures efficient execution and cost-effectiveness.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Adjusting Spark Configurations

# COMMAND ----------

# Set Spark configurations for optimal resource usage and performance

# Set the number of shuffle partitions to 200
# 'spark.sql.shuffle.partitions' controls the number of partitions created when shuffling data
# A value of 200 is commonly used to balance parallelism and resource usage for large datasets
# Increasing or decreasing this value can help optimize performance, depending on cluster size and data volume
spark.conf.set("spark.sql.shuffle.partitions", "200")

# Set the amount of memory allocated to each Spark executor to 8 GB
# 'spark.executor.memory' defines the memory available for each executor running on the cluster
# Allocating 8 GB of memory per executor helps handle large data processing and model training tasks
# Adjust this value based on the size of your data and the available memory on your cluster
spark.conf.set("spark.executor.memory", "8g")

# Set the number of CPU cores allocated to each executor to 4
# 'spark.executor.cores' specifies the number of CPU cores assigned to each executor
# Allocating 4 cores allows each executor to process multiple tasks concurrently, improving performance
# Adjusting this setting helps balance resource utilization across the cluster, depending on workload requirements
spark.conf.set("spark.executor.cores", "4")


# COMMAND ----------

# MAGIC %md
# MAGIC ##Choosing the Right Cluster
# MAGIC * **Cluster Size**: Select a cluster with sufficient memory and CPU cores to handle the dataset size.
# MAGIC * **Autoscaling**: Enable autoscaling to adjust resources based on workload demand.

# COMMAND ----------

# MAGIC %md
# MAGIC #Model Deployment and Serving
# MAGIC Deploy the trained model for real-time inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Registering the Model with MLflow

# COMMAND ----------

# Register the trained model with MLflow Model Registry

# Construct the URI for the trained model
# 'mlflow.active_run().info.run_id' gets the ID of the current MLflow run
# 'runs:/{}/model' specifies the path to the model artifact from the current run
# This URI points to the model that was logged in the active MLflow run
model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)

# Register the model in the MLflow Model Registry under the specified name
# 'mlflow.register_model()' adds the model to the Model Registry, making it available for tracking, versioning, and deployment
# 'model_uri' is the location of the trained model in the current run
# 'NYCTaxiTripDurationModel' is the name under which the model is registered in the Model Registry
# If a model with this name already exists, a new version will be created
model_details = mlflow.register_model(model_uri, "NYCTaxiTripDurationModel")


# COMMAND ----------

# MAGIC %md
# MAGIC #Transitioning the Model to Production

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Initialize an MLflow client to interact with the MLflow Model Registry
# 'MlflowClient()' provides programmatic access to manage models, experiments, and runs in MLflow
client = MlflowClient()

# Transition the specified model version to the "Production" stage in the Model Registry
# 'name' specifies the registered model name ('NYCTaxiTripDurationModel')
# 'version' specifies the model version obtained from 'model_details.version', representing the latest registered version
# 'stage' is set to "Production", indicating that this model version is now deployed for production use
client.transition_model_version_stage(
    name="NYCTaxiTripDurationModel",
    version=model_details.version,
    stage="Production"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Serving the Model
# MAGIC * **Enable Model Serving**: In Databricks, navigate to the registered model and enable serving.
# MAGIC * **Create an Endpoint**: Configure an endpoint for real-time predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Making Predictions

# COMMAND ----------

import pandas as pd
import mlflow.pyfunc

# Load the production model from the MLflow Model Registry
# 'models:/NYCTaxiTripDurationModel/Production' specifies the registered model and stage
# 'models:/' indicates loading from the Model Registry, while 'NYCTaxiTripDurationModel' is the model name
# 'Production' specifies that the latest version in the Production stage should be loaded
model = mlflow.pyfunc.load_model("models:/NYCTaxiTripDurationModel/Production")

# Prepare input data for making predictions
# The input data is created as a pandas DataFrame with the same structure as the training data
# The features include 'trip_distance', 'passenger_count', 'pickup_hour', 'PULocationID', 'DOLocationID', and 'pickup_day'
# This ensures the input data is aligned with the model's expected feature set
input_data = pd.DataFrame({
    'trip_distance': [3.5],    # Example trip distance in miles
    'passenger_count': [2],    # Number of passengers
    'pickup_hour': [17],       # Hour of pickup (5 PM)
    'PULocationID': [132],     # Pickup location ID
    'DOLocationID': [158],     # Drop-off location ID
    'pickup_day': [4]          # Day of the week (e.g., 4 = Wednesday)
})

# Ensure the same preprocessing steps are applied to the input data
# Define a function to apply the necessary transformations to the input data
# For simplicity, assume we have a function 'preprocess_input' that performs encoding, feature assembly, etc.
# This function should match the preprocessing steps used during model training
def preprocess_input(df):
    # Apply the same transformations as during training
    # (e.g., encoding categorical variables, feature vector assembly)
    # This step ensures consistency between training and inference
    # In practice, this function should include the actual transformations applied to the input data
    # For now, we return the DataFrame as is (placeholder)
    return df_prepared

# Apply preprocessing to the input data
# This prepares the data in the same way it was prepared during training
input_data_prepared = preprocess_input(input_data)

# Make predictions using the loaded model
# 'predict()' generates predictions for the prepared input data
# The prediction output is a single value representing the predicted trip duration in minutes
prediction = model.predict(input_data_prepared)

# Print the predicted trip duration, formatted to two decimal places
print(f"Predicted trip duration: {prediction[0]:.2f} minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC #Monitoring and Maintaining the Model
# MAGIC Continuous monitoring ensures the model remains accurate over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Performance Monitoring

# COMMAND ----------

# Function to monitor and log model performance
def monitor_model_performance(actuals, predictions):
    # Calculate the Root Mean Squared Error (RMSE) between actuals and predictions
    # 'evaluator' is an instance of RegressionEvaluator initialized to measure RMSE
    # It compares the actual target values with the model's predicted values
    rmse = evaluator.evaluate(predictions)
    
    # Log the RMSE metric to MLflow
    # This records the RMSE in the MLflow tracking server, enabling tracking of model performance over time
    mlflow.log_metric("rmse", rmse)
    
    # Print the RMSE to provide immediate feedback on the model's performance
    print(f"Logged RMSE: {rmse:.2f} minutes")



# COMMAND ----------

# MAGIC %md
# MAGIC ##Data Drift Detection
# MAGIC * **Regular Retraining**: Schedule periodic retraining if data patterns change.
# MAGIC * **Alerting Mechanisms**: Implement alerts for significant performance degradation.

# COMMAND ----------

# MAGIC %md
# MAGIC #Best Practices for ML at Scale
# MAGIC * **Data Sampling for Prototyping**: Use a subset of data for initial development to speed up iterations.
# MAGIC * **Efficient Data Formats**: Utilize Parquet or Delta Lake for faster I/O operations.
# MAGIC * **Parallelism**: Leverage Spark's distributed processing by increasing parallelism where appropriate.
# MAGIC * **Lazy Evaluation** : Remember that Spark operations are lazy; use count() or collect() judiciously to trigger actions.
# MAGIC * **Broadcast Variables**: Use broadcast variables for small lookup tables to optimize joins.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this section, you learned how to use Spark at scale within the Databricks environment to tackle machine learning tasks on a large dataset. By leveraging Spark’s distributed computing capabilities, you were able to process millions of records efficiently, perform feature engineering, train a machine learning model, and optimize hyperparameters. By integrating MLflow, you could easily track experiments and manage the model lifecycle.
# MAGIC
# MAGIC Using the NYC Yellow Taxi dataset, you explored practical steps and best practices for managing large-scale machine learning projects, from data ingestion to deployment. These techniques empower you, as a data scientist or engineer, to build robust, scalable models that can deliver valuable insights and predictions in real-world applications.

# COMMAND ----------



# Train a Random Forest regressor
rf = RandomForestRegressor(featuresCol='features', labelCol='label', numTrees=50, maxDepth=10)
rf_model = rf.fit(train_data)

# Make predictions on the test set
predictions = rf_model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f"Test RMSE = {rmse:.2f} minutes")


# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials

# Define the search space
search_space = {
    'numTrees': hp.choice('numTrees', [20, 30, 40, 50, 60, 70, 80, 90, 100]),
    'maxDepth': hp.choice('maxDepth', list(range(5, 16)))
}

def objective_function(params):
    num_trees = int(params['numTrees'])
    max_depth = int(params['maxDepth'])

    # Your model training logic here
    loss = 1.0  # Example loss calculation

    return {'loss': loss, 'status': 'ok'}

# Use SparkTrials for distributed hyperparameter tuning
spark_trials = SparkTrials(parallelism=4)

# Run Hyperopt
best_params = fmin(
    fn=objective_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=spark_trials
)

print("Best parameters:", best_params)


# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a parameter grid around the best parameters found
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(best_params['numTrees']), int(best_params['numTrees']) + 20]) \
    .addGrid(rf.maxDepth, [int(best_params['maxDepth']), int(best_params['maxDepth']) + 2]) \
    .build()

# Set up cross-validation
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cv_model = cv.fit(train_data)

# Evaluate the best model
best_model = cv_model.bestModel
predictions = best_model.transform(test_data)
rmse = evaluator.evaluate(predictions)
print(f"Best Model Test RMSE: {rmse:.2f} minutes")


# COMMAND ----------

import mlflow
import mlflow.spark

with mlflow.start_run():
    # Train the model with best hyperparameters
    rf = RandomForestRegressor(featuresCol='features', labelCol='label', numTrees=int(best_params['numTrees']), maxDepth=int(best_params['maxDepth']))
    rf_model = rf.fit(train_data)
    
    # Make predictions and evaluate
    predictions = rf_model.transform(test_data)
    rmse = evaluator.evaluate(predictions)
    
    # Log parameters and metrics
    mlflow.log_param("numTrees", int(best_params['numTrees']))
    mlflow.log_param("maxDepth", int(best_params['maxDepth']))
    mlflow.log_metric("rmse", rmse)
    
    # Log the model
    mlflow.spark.log_model(rf_model, "random-forest-model")
    
    print(f"Model logged with RMSE: {rmse:.2f} minutes")

