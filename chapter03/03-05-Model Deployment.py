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
# MAGIC # Setup the  run_id and the model name

# COMMAND ----------

# The unique identifier for the MLflow run where the model was logged.
# This ID corresponds to the specific experiment run in MLflow.
run_id = "9410278920a74076a86803cddd3e982e"  

# This is the name that we give our model
model_name = "hotel_booking_cancel_prediction"  

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model

# COMMAND ----------

# Register the model in MLflow's Model Registry.
# The 'register_model()' function allows you to take a model that was logged during a run 
# and register it in the MLflow Model Registry under a specific name.
# This is useful for tracking versions of models and managing their lifecycle (e.g., staging, production).

# 'runs:/{run_id}/model' specifies the model's location in the MLflow tracking system.
# It points to the model artifact saved during a specific run identified by 'run_id'.
# 'run_id' is the unique identifier for the MLflow run where the model was trained and logged.
# '/model' is the subpath to the model within that run.

# 'model_name' is the name you want to use to register the model in the MLflow Model Registry.
# This allows you to track different versions of the model under this name.

model_version = mlflow.register_model(f'runs:/{run_id}/model', model_name)

# 'model_version' will store the version number of the newly registered model.
# MLflow automatically tracks versions, so each time a new model is registered with the same name, 
# it increments the version number.

# COMMAND ----------

# Create an instance of the MlflowClient.
# This client will be used to interact with the MLflow Model Registry to perform operations
# like transitioning a model to different stages (e.g., Staging, Production).
client = MlflowClient()

# 'search_model_versions()' is a method provided by MlflowClient that returns details about
# all the versions of a registered model. It queries the MLflow Model Registry for versions
# of the model with the specified name.

# 'model_name' is the name of the model that you want to search for in the Model Registry.
# The f-string (f"name='{model_name}'") is used to dynamically insert the model's name into the search query.
# This query will retrieve all the registered versions for the model with the given name.
versions = client.search_model_versions(f"name='{model_name}'")

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

# Transition the specified version of a registered model to a new stage in the MLflow Model Registry.
# The stage can represent the lifecycle phase of the model, such as 'Staging', 'Production', or 'Archived'.

# 'client' is an instance of the MlflowClient, which allows programmatic interaction with MLflow's Model Registry.

client.transition_model_version_stage(
    # The name of the model in the Model Registry. This should match the model's name 
    # that was used when the model was first registered (e.g., 'Logistic_Regression_Model').
    name = model_name,       

    # The version number of the model you want to transition. 
    # Each model can have multiple versions, and this specifies which one to transition.
    # In this case, we're transitioning version 1.                             
    version = "1",           

    # The target stage to which you want to transition the model. 
    # Common stages include:
    # - 'Production': Indicates the model is ready for production use.
    # - 'Staging': For models undergoing evaluation before production.
    # - 'Archived': For older versions of the model that are no longer in use.
    stage = 'Production'     
)
