# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px; width:30%; height: auto;" src="https://i.imgur.com/FWzhbhX.jpeg"   />   
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 05-7-Feature Lookup
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      12-24-2024
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
# MAGIC ##Make sure to run the notebook with our constants

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

# COMMAND ----------

# MAGIC %run "../common/Constants"

# COMMAND ----------

# MAGIC %md
# MAGIC #Perform Feature Lookup

# COMMAND ----------

# Initialize Feature Store client
fs = FeatureStoreClient()

FEATURE_TABLE_NAME = "customer_transaction_features"
# Define the fully qualified feature table name (Unity Catalog)
feature_table_name = f"{CATALOG_NAME}.{FEATURE_STORE_DB}.{FEATURE_TABLE_NAME}"

# COMMAND ----------

# Simulate a new batch of data for scoring
batch_data = spark.createDataFrame([
    ("CUST8488", "2024-12-10", 100.0, "Purchase"),
    ("CUST25878", "2024-12-11", 200.0, "Transfer"),
], ["Customer_ID", "Date", "Transaction_Amount", "Transaction_Type"])

# Define the feature lookups
feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        feature_names=["TotalTransactionAmount", "TransactionCount", "AverageTransactionAmount"],
        lookup_key="Customer_ID"
    )
]

# COMMAND ----------

feature_lookups

# COMMAND ----------

# Retrieve features using the Feature Store
training_set = fs.create_training_set(
    df=batch_data,
    feature_lookups=feature_lookups,
    label=None  # Set `label` if the dataset contains a label column for training
)

# Load the resulting DataFrame
scoring_data = training_set.load_df()

# Display the resulting DataFrame with joined features
scoring_data.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Visualize Feature Distributions

# COMMAND ----------

import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas for visualization
scoring_data_pd = scoring_data.toPandas()

# Plot histograms for numerical features
features = ["TotalTransactionAmount", "TransactionCount", "AverageTransactionamount"]

plt.figure(figsize=(12, 6))
for i, feature in enumerate(features, 1):
    plt.subplot(1, len(features), i)
    plt.hist(scoring_data_pd[feature], bins=20, alpha=0.7, color='blue')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

