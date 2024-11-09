# Databricks notebook source
import numpy as np
import pandas as pd

from databricks.feature_store import FeatureStoreClient

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructType, StructField
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array



# COMMAND ----------

# MAGIC %md
# MAGIC # Dataset Information
# MAGIC I am using the **MovieLens 1M Dataset**.
# MAGIC It can be downloaded (here)[https://grouplens.org/datasets/movielens/1m/]
# MAGIC This dataset contains 1 Million Movie Ratings from 6000 users on 4000 Movies. The dataset was released on 2/2/2003

# COMMAND ----------

# MAGIC %md
# MAGIC #Read the source data 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Read in the Ratings Dataset

# COMMAND ----------

# Define schema explicitly
ratings_schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("movie_id", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", LongType(), True)
    # Add more columns as needed
])

ratings_df = spark.read.option("sep", "::").option("header", "false") \
    .csv("dbfs:/FileStore/datasets/MovieLens-1M/ratings.dat", schema=ratings_schema)

# COMMAND ----------

ratings_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Read in the Movies Dataset

# COMMAND ----------

# Define schema explicitly
movies_schema = StructType([
    StructField("movie_id", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True)
    # Add more columns as needed
])

movies_df = spark.read.csv("dbfs:/FileStore/datasets/MovieLens-1M/ratings.dat", header=False, schema=movies_schema)
movies_df = spark.read.option("sep", "::").option("header", "false") \
    .csv("dbfs:/FileStore/datasets/MovieLens-1M/movies.dat", schema=movies_schema)

# Split Genres by pipe to create an array of genres
movies_df = movies_df.withColumn("Genres", F.split(movies_df["Genres"], "\|"))    


# COMMAND ----------

movies_data_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Read in the Users dataset

# COMMAND ----------

# Define the schema explicitly
users_schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("occupation", IntegerType(), True),
    StructField("zip_code", IntegerType(), True)
])

users_df = spark.read.option("sep", "::").option("header", "false") \
    .csv("dbfs:/FileStore/datasets/MovieLens-1M/users.dat", schema=users_schema)

# COMMAND ----------

users_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merge the movies_df, ratings_df and users_df

# COMMAND ----------

# First, join ratings_df and movies_df on "movie_id"
merged_df = ratings_df.join(movies_df, on="movie_id", how="inner")

# Then, join the resulting DataFrame with users_df on "user_id"
merged_df = merged_df.join(users_df, on="user_id", how="inner")

# Show the results
merged_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create User Features

# COMMAND ----------

# Group by 'user_id' and calculate average rating and rating count
user_features = merged_df.groupBy("user_id").agg(
    F.avg("rating").alias("avg_rating"),
    F.count("rating").alias("rating_count")
)

# Show the result
user_features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create Movie Features

# COMMAND ----------

movie_features = merged_df.groupBy("movie_id").agg(
    F.avg("rating").alias("avg_movie_rating"),
    F.count("rating").alias("movie_rating_count")
)

# Show the result
movie_features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Normalize the User and Movie Features for Consistency

# COMMAND ----------

# MAGIC %md
# MAGIC ##Normalize the User Features

# COMMAND ----------

# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=["avg_rating", "rating_count"], outputCol="user_features")
user_features = assembler.transform(user_features)

# Scale the features
scaler = StandardScaler(inputCol="user_features", outputCol="scaled_user_features", withMean=True, withStd=True)
scaler_model = scaler.fit(user_features)
user_features_scaled = scaler_model.transform(user_features)

# Convert the scaled vector column to an array
user_features_scaled = user_features_scaled.withColumn("scaled_user_features_array", vector_to_array("scaled_user_features"))

# Split the array into separate columns
user_features_scaled = user_features_scaled.select(
    "user_id",
    col("scaled_user_features_array")[0].alias("scaled_avg_rating"),
    col("scaled_user_features_array")[1].alias("scaled_rating_count")
)

# Show the scaled results
user_features_scaled.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Normalize the Movie Features

# COMMAND ----------

# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=["avg_movie_rating", "movie_rating_count"], outputCol="movie_features")
movie_features = assembler.transform(movie_features)

# Scale the features
scaler = StandardScaler(inputCol="movie_features", outputCol="scaled_movie_features", withMean=True, withStd=True)
scaler_model = scaler.fit(movie_features)
movie_features_scaled = scaler_model.transform(movie_features)

# Convert the scaled vector column to an array
movie_features_scaled = movie_features_scaled.withColumn("scaled_movie_features_array", vector_to_array("scaled_movie_features"))

# Split the array into separate columns
movie_features_scaled = movie_features_scaled.select(
    "movie_id",
    F.col("scaled_movie_features_array")[0].alias("scaled_avg_rating"),
    F.col("scaled_movie_features_array")[1].alias("scaled_rating_count")
)

# Show the scaled results
user_features_scaled.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #Write to the the Databricks Feature Store

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create the Database

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS book_ai_ml_lakehouse.movielens_features_db;

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG book_ai_ml_lakehouse;
# MAGIC
# MAGIC SHOW DATABASES;

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create the user_features table

# COMMAND ----------

# Initialize Feature Store client
fs = FeatureStoreClient()

# Create the user_features table
fs.create_table(
    name='movielens_features_db.user_features',
    primary_keys='user_id',
    df=user_features_scaled,
    description='User features (scaled) from MovieLens dataset'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Creaet the movie_features table

# COMMAND ----------

# Create the move_features table
fs.create_table(
    name='movielens_features_db.movie_features',
    primary_keys='movie_id',
    df=movie_features_scaled,
    description='Movie features from MovieLens dataset'
)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------


