# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px;" src="https://learning.oreilly.com/covers/urn:orm:book:9781098139711/400w/" />  
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 03-04-Model Deployment
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-13-2024
# MAGIC
# MAGIC    Purpose:   This notebook registers a model in the Model Registry
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Setup the run_id and the model name
# MAGIC         2 - Register the model
# MAGIC         3 - Move the model into production
# MAGIC         

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC #Prepare our Environment for model registration in Unity

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Unit Catalog and Schema

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Make sure the Catalog is created
# MAGIC CREATE CATALOG IF NOT EXISTS `book_ai_ml_lakehouse`;
# MAGIC
# MAGIC -- Create the schema for the models
# MAGIC CREATE SCHEMA IF NOT EXISTS book_ai_ml_lakehouse.registered_models;
# MAGIC
# MAGIC -- Make sure we can see the newly created schema
# MAGIC SHOW SCHEMAS IN book_ai_ml_lakehouse

# COMMAND ----------

# MAGIC %md
# MAGIC ##Setup the  run_id and the Unity Model name

# COMMAND ----------

# The unique identifier for the MLflow run where the model was logged.
# This ID corresponds to the specific experiment run in MLflow.
run_id = "06ab7ac6a53c4431a0db34827220ddf6"  

# The name assigned to the model. This will be used as a reference for the registered model.
model_name = "hotel_booking_cancel_prediction"  

# The catalog name in Unity Catalog, representing the top-level namespace.
catalog_name = "book_ai_ml_lakehouse"

# The schema name within the catalog, typically used for organizing registered models.
schema_name = "registered_models"

# Construct the full Unity Catalog model name, including catalog, schema, and model name.
unity_model_name = f"{catalog_name}.{schema_name}.{model_name}"

# Print the full model location in Unity Catalog.
print(f"Unity Model Name: {unity_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Make sure that we use the Unity Catalog for our registry

# COMMAND ----------

# Set the registry URI to Databricks Unity Catalog.
# This tells MLflow to use Unity Catalog for managing the model registry.
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model

# COMMAND ----------

# Create an instance of the MlflowClient.
# This client is used to interact with the MLflow Model Registry,
# enabling operations such as registering models, managing versions, and transitioning their lifecycle stages.
client = MlflowClient()

# Register the model in MLflow's Model Registry within Unity Catalog.
# The 'register_model()' function associates a model artifact, logged during a specific MLflow run,
# with a registered model name in the Unity Catalog. Unity Catalog provides centralized access control,
# auditing, and lineage for models.

# 'runs:/{run_id}/model' specifies the path to the model artifact:
# - 'run_id' is the unique identifier for the MLflow run where the model was logged.
# - '/model' is the subpath indicating the location of the logged model artifact within the run.

# 'unity_model_name' is the name under which the model will be registered in the Unity Catalog Model Registry.
# This name is unique within the catalog and helps track multiple versions of the model.

# The model will be registered in Unity Catalog, ensuring benefits such as centralized governance,
# lineage tracking, and compatibility with the open-source MLflow client.
model_version = mlflow.register_model(f'runs:/{run_id}/model', unity_model_name)

# 'model_version' holds the metadata of the newly registered model version, including the version number.
# Each time a model with the same name is registered, MLflow automatically increments the version number,
# enabling version control and lifecycle management.

# COMMAND ----------

# MAGIC %md
# MAGIC #Examine all model versions in the Registry

# COMMAND ----------


# 'search_model_versions()' is a method provided by MlflowClient that returns details about
# all the versions of a registered model. It queries the MLflow Model Registry for versions
# of the model with the specified name.

# 'model_name' is the name of the model that you want to search for in the Model Registry.
# The f-string (f"name='{model_name}'") is used to dynamically insert the model's name into the search query.
# This query will retrieve all the registered versions for the model with the given name.
versions = client.search_model_versions(f"name='{unity_model_name}'")

# The 'versions' variable will contain a list of model version metadata, where each entry in the list
# contains information about a specific version of the model, such as version number, stage (e.g., Production, Staging),
# and other details like creation time and run ID.

# You can iterate over 'versions' to access or print the metadata of each model version.
for version in versions:
    print(f"Model Name: {version.name}, Version: {version.version}, Stage: {version.current_stage}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Move the model into production

# COMMAND ----------

# create "Champion" alias for version 1 of model "prod.ml_team.iris_model"
client.set_registered_model_alias(unity_model_name, "Champion", 1)

# COMMAND ----------

# MAGIC %md
# MAGIC #End of Notebook
