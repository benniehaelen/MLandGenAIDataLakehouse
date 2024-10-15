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

# Import the 'os' module, which provides a way to interact with the operating system.
# The 'os' module allows you to set and retrieve environment variables, among other functionalities.
import os

# Set an environment variable 'DATABRICKS_TOKEN' with the value of your Databricks personal access token.
# This token will be used for authenticating API requests to Databricks services (e.g., for model serving or running jobs).
# The 'os.environ' dictionary allows you to set or access environment variables in the current session.


# COMMAND ----------


