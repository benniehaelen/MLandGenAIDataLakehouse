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
# MAGIC ##Make sure to run the notebook with our constants

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %run "../common/Constants"

# COMMAND ----------

# MAGIC %md
# MAGIC #Read in the Features from the Feature Table

# COMMAND ----------

# Initialize Feature Store client
fs = FeatureStoreClient()

FEATURE_TABLE_NAME = "customer_transaction_features"
# Define the fully qualified feature table name (Unity Catalog)
feature_table_name = f"{CATALOG_NAME}.{FEATURE_STORE_DB}.{FEATURE_TABLE_NAME}"

# Read features from the feature store
customer_features = fs.read_table(name=feature_table_name)

# Convert to Pandas DataFrame
customer_features_pd = customer_features.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #Build the model

# COMMAND ----------

# Simulate a churn target column
customer_features_pd["Churn"] = (customer_features_pd["TotalTransactionAmount"] < 500).astype(int)

# Split data into features and target
X = customer_features_pd.drop(columns=["Customer_ID", "Churn"])
y = customer_features_pd["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluate the model

# COMMAND ----------

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #End of Notebook
