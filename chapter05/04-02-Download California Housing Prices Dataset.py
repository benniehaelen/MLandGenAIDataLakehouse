# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px;" src="https://learning.oreilly.com/covers/urn:orm:book:9781098139711/400w/" />  
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 04-01-Download Calfiornia Housing Prices Dataset
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-24-2024
# MAGIC
# MAGIC    Purpose:   This notebook downloads the California Housing Prices dataset from the Kaggle Website, and saves it as a Unity Table
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Handle the Pre-Requisties
# MAGIC            1-1 - Make sure kaggle and kagglehub are installed
# MAGIC         2 - Use KaggleHub to download the Kaggle Dataset
# MAGIC            2-1 - Download the dataset to a local path
# MAGIC            2-2 - Copy the local file to our DBFS datasets location
# MAGIC         3 - Prepare and Save the Dataset
# MAGIC            3-1 - Read the source file from our dbfs location
# MAGIC            3-2 - Create our Catalog and Schema (if needed)
# MAGIC            3-3 - Save our Dataframe as a Delta Table in our Catalog
# MAGIC               
# MAGIC

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

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ##Make sure to run the notebook with our constants

# COMMAND ----------

# MAGIC %run "../common/Constants"

# COMMAND ----------

# File locations
CA_HOUSING_PRICES_LOCAL_FILE_NAME = "housing.csv"
KAGGLE_FILE_LOCATION = "kallolnath1/california-housing-prices-dataset"

# Unity Table Name
TABLE_NAME   = "ca_housing_prices"

# COMMAND ----------

# MAGIC %md
# MAGIC #Use KaggleHub to download the Kaggle Dataset
# MAGIC [Link to the dataset](https://www.kaggle.com/datasets/kallolnath1/california-housing-prices-dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Downlaod the dataset to a local path

# COMMAND ----------

# Import the 'kagglehub' module to interact with Kaggle datasets.
import kagglehub  

# Download the latest version of the specified dataset.
# 'dataset_download' takes the dataset identifier as an argument.
# In this case, it downloads the dataset 'test-file' by the user 'kallolnath1',
# which is the Kaggle Dataset
local_path = kagglehub.dataset_download(KAGGLE_FILE_LOCATION)

# Print the local file path where the dataset files have been downloaded.
print("Path to dataset files:", local_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Copy the local file to our DBFS datasets location

# COMMAND ----------

import shutil

# Construct the full local path by appending the file name to the existing local directory path
local_path = f"{local_path}/{CA_HOUSING_PRICES_LOCAL_FILE_NAME}"

# Print the local path to verify correctness
print(f"The file has been downloaded to local path: {local_path}")  

# Define the DBFS path where you want to move the file
# This path specifies where the file will be stored in the Databricks File System (DBFS)
dbfs_path = f"{DBFS_DATASET_DIRECTORY}/{CA_HOUSING_PRICES_LOCAL_FILE_NAME}"
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

# Display the first 5 rows of the DataFrame to verify successful loading
df.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Convert the columns to the correct data types

# COMMAND ----------

# Type conversions
df = df.withColumn("longitude", col("longitude").cast(FloatType()))
df = df.withColumn("latitude", col("latitude").cast(FloatType()))
df = df.withColumn("housing_median_age", col("housing_median_age").cast(IntegerType()))
df = df.withColumn("total_rooms", col("total_rooms").cast(IntegerType()))
df = df.withColumn("total_bedrooms", col("total_bedrooms").cast(IntegerType()))
df = df.withColumn("population", col("population").cast(IntegerType()))
df = df.withColumn("households", col("households").cast(IntegerType()))
df = df.withColumn("median_income", col("median_income").cast(FloatType()))
df = df.withColumn("median_house_value", col("median_house_value").cast(FloatType()))
df = df.withColumn("ocean_proximity", col("ocean_proximity").cast(StringType()))

df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create our Catalog and Schema (if needed)

# COMMAND ----------

# Create a new catalog in Unity Catalog using Spark SQL
# The 'CREATE CATALOG IF NOT EXISTS' command ensures the catalog is created only if it doesn't already exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
print(f"Catalog '{CATALOG_NAME}' created.")  # Print confirmation of catalog creation

# Create a new schema (or database) within the specified catalog using Spark SQL
# The 'CREATE SCHEMA IF NOT EXISTS' command ensures the schema is created only if it doesn't already exist
# The schema is created within the specified catalog to organize tables under a common namespace
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

# Print confirmation of schema creation
print(f"Schema '{SCHEMA_NAME}' created in catalog '{CATALOG_NAME}'.")  

# COMMAND ----------

# MAGIC %md
# MAGIC ##Save our Dataframe as a Delta Table in our Catalog

# COMMAND ----------

# Combine catalog, schema, and table names to create the fully qualified table path
# This format ensures that the table is saved under the specified catalog and schema in Unity Catalog
full_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"

# Save the DataFrame as a Delta table in Unity Catalog using the specified full table name
# 'format("delta")' specifies that the data should be stored in Delta format
# 'mode("overwrite")' ensures that if a table with the same name already exists, it will be replaced
df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)

# Print a confirmation message indicating the table has been saved in Unity Catalog
print(f"Table '{full_table_name}' saved in Unity Catalog.")

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook Complete
