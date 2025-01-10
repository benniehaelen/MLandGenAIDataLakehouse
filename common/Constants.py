# Databricks notebook source
#
# Root directory for the book's datasets
#

# This is the Root Directory for all of our datasets
DBFS_DATASET_DIRECTORY = "/dbfs/FileStore/datasets"

# NYC Taxi Dataset DBFS path
NYC_TAXI_DATASET_PATH = f"/FileStore/datasets/nyc_taxi_data/"

# NYC Taxi Delta Dataset DBFS path
DELTA_NYC_DATASET_PATH = f"/FileStore/datasets/delta_nyc_taxi_data"


#
# UNITY Constants
#

# Book UNITY catalog name
CATALOG_NAME = "book_ai_ml_lakehouse"

# Book UNITFY Schema name 
SCHEMA_NAME  = "automl"

FEATURE_STORE_DB = "feature_store_db"
