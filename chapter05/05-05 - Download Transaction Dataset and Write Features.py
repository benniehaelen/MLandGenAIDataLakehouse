# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px; width:30%; height: auto;" src="https://i.imgur.com/FWzhbhX.jpeg"   />  
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 05-05-Donwload Transaction Dataset and Write Features
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      12-23-2024
# MAGIC
# MAGIC    Purpose:   This notebook will read the customer transaction analysis dataset from Kaggle and transform the data into features
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Read the Delta table witeh the housing prices
# MAGIC         2 - Start the modeling phase
# MAGIC             2-1 - Perform a train/test split of the data
# MAGIC             2-2 - Investigate the Shape of the datasets
# MAGIC             2-3 - Convert our training Pandas Dataframe to Spark
# MAGIC             2-4 - Start the AutoML Regression
# MAGIC         3 - Study the results of the regression and make predictions
# MAGIC             3-1 - Retrieve the URI of the best model
# MAGIC             3-2 - Create the Test Features
# MAGIC             3-3 - Load the best model from the MLflow function
# MAGIC             3-4 - Use the model to make prediction
# MAGIC             3-5 - Combine predictions and actual
# MAGIC             3-6 - Create a plot comparing the actuals with the predictions
# MAGIC             3-7 - Create a joint plot of actual vs predicted

# COMMAND ----------

# MAGIC %md
# MAGIC #Handle Pre-Requisites

# COMMAND ----------

# MAGIC %md
# MAGIC ##Make sure that kaggle and kagglehub are installed

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

# MAGIC %pip install kagglehub

# COMMAND ----------

# MAGIC %md
# MAGIC ##Make sure to run the notebook with our constants

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, count, col, sum as _sum, avg, countDistinct
from pyspark.sql.types import IntegerType, FloatType

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

# MAGIC %run "../common/Constants"

# COMMAND ----------

# MAGIC %md
# MAGIC #Use KaggleHub to download the Kaggle Dataset
# MAGIC [Link to the dataset](https://www.kaggle.com/datasets/sharmadeepankaj/customer-transaction-analysis-dataset)

# COMMAND ----------

# File locations
CUSTOMER_TRANSACTIONS_LOCAL_FILE_NAME = "finance_dataset.csv"
KAGGLE_FILE_LOCATION    = "sharmadeepankaj/customer-transaction-analysis-dataset"

# Table Name
TABLE_NAME   = "customer_transactions"

# COMMAND ----------

# Import the 'kagglehub' module to interact with Kaggle datasets.
import kagglehub

# Download the latest version of the specified dataset.
# 'dataset_download' takes the dataset identifier as an argument.
# In this case, it downloads the dataset 'tcustomer-transaction-analysis-dataset' by the user 'sharmadeepankaj',
# which is the Customer Transaction Dataset
local_path = kagglehub.dataset_download(KAGGLE_FILE_LOCATION)

# Print the local file path where the dataset files have been downloaded.
print("Path to dataset files:", local_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Copy the local file to our DBFS datasets location

# COMMAND ----------

import shutil

# Construct the full local path by appending the file name to the existing local directory path
local_path = f"{local_path}/{CUSTOMER_TRANSACTIONS_LOCAL_FILE_NAME}"

# Print the local path to verify correctness
print(f"The file has been downloaded to local path: {local_path}")  

# Define the DBFS path where you want to move the file
# This path specifies where the file will be stored in the Databricks File System (DBFS)
dbfs_path = f"{DBFS_DATASET_DIRECTORY}/{CUSTOMER_TRANSACTIONS_LOCAL_FILE_NAME}"
print(f"The file will be copied to the dfbs location: {dbfs_path}")

# Use shutil.copy() to move the file from the local path to the DBFS path
# This function copies the file to the specified DBFS directory, making it accessible to Databricks
shutil.copy(local_path, dbfs_path)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/FileStore/datasets

# COMMAND ----------

# MAGIC %md
# MAGIC #Prepare and Save the Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ##Read the source file from our dbfs location

# COMMAND ----------

# Check if the path starts with '/dbfs'
# The '/dbfs' prefix is used for local file system access, but Spark needs the path in DBFS format
if dbfs_path.startswith("/dbfs"):
    dbfs_path = dbfs_path[5:]  # Remove the first 5 characters to strip the '/dbfs' prefix

# Print the adjusted DBFS path to verify it has been modified correctly
print(f"Adjusted DBFS path: {dbfs_path}")

# Read the CSV file from the adjusted DBFS path using Spark
# The 'header=True' option specifies that the first row of the file contains column names
df = spark.read.csv(dbfs_path, header=True)

# Print out the schema
df.printSchema()
# Display the first 5 rows of the DataFrame to verify successful loading
df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Filter out all the transactions that are not "Completed"

# COMMAND ----------

# Filter for completed transactions
valid_transactions = df.filter(col("Transaction_Status") == "Completed")


# COMMAND ----------

# MAGIC %md
# MAGIC #Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ##Aggregate by CustomerID

# COMMAND ----------

# Aggregate features
customer_features = (
    valid_transactions.groupBy("Customer_ID")
    .agg(
        _sum("Transaction_Amount").alias("TotalTransactionAmount"),
        countDistinct("ID").alias("TransactionCount"),
        avg("Transaction_Amount").alias("AverageTransactionAmount"),
        countDistinct("Category").alias("CategoryDiversity")
    )
)

# Display customer-level features
customer_features.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Write the features to the feature Store
# MAGIC Register the aggregated features into the Databricks Feature Store.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create our "feature_store_db" schema to hold our feature databases

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS book_ai_ml_lakehouse.feature_store_db;

# COMMAND ----------

# MAGIC %md
# MAGIC ##Write our customer_transaction_features table

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# Initialize Feature Store client
fs = FeatureStoreClient()

FEATURE_TABLE_NAME = "customer_transaction_features"
# Define the fully qualified feature table name (Unity Catalog)
feature_table_name = f"{CATALOG_NAME}.{FEATURE_STORE_DB}.{FEATURE_TABLE_NAME}"

# Create the feature table. Note that Customer_ID will be the primary
# key of our table
fs.create_table(
    name=feature_table_name,
    primary_keys=["Customer_ID"],  # Primary key for the table
    schema=customer_features.schema,  # Schema from Spark DataFrame
    description="Features derived from Kaggle's customer transaction analysis dataset"
)

# Write features to the Feature Store
fs.write_table(
    name=feature_table_name,
    df=customer_features,  # Spark DataFrame with features
    mode="overwrite"  # Write mode
)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Add Custom Tags to our Features Table

# COMMAND ----------

# Add tags for governance and discovery
# Add tags for governance and discovery
fs.set_feature_table_tag(
    table_name=feature_table_name,
    key="source",
    value="Kaggle customer transaction dataset"
)
fs.set_feature_table_tag(
    table_name=feature_table_name,
    key="owner",
    value="data-science-team"
)
fs.set_feature_table_tag(
    table_name=feature_table_name,
    key="sensitivity",
    value="low"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #End of Notebook
