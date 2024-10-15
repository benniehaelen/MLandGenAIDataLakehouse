# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
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

# MAGIC %pip install azure-identity azure-keyvault-secrets

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Import the 'os' module, which provides a way to interact with the operating system.
# The 'os' module allows you to set and retrieve environment variables, among other functionalities.
import os

# Set an environment variable 'DATABRICKS_TOKEN' with the value of your Databricks personal access token.
# This token will be used for authenticating API requests to Databricks services (e.g., for model serving or running jobs).
# The 'os.environ' dictionary allows you to set or access environment variables in the current session.
# Replace with your Key Vault URL

os.environ['DATABRICKS_TOKEN']  =''

# Note: This is a sensitive access token, so make sure you do not expose it in your code.
# In a production environment, it's recommended to use more secure ways to manage sensitive information,
# such as using environment variable management tools or secrets management services (e.g., AWS Secrets Manager, Azure Key Vault).

# COMMAND ----------

import json
import mlflow
import requests
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# Define the path to your Delta table
delta_file = "dbfs:/FileStore/datasets/hotel_bookings.delta"

# Read the Delta table as a Spark DataFrame
spark_df = spark.read.format("delta").load(delta_file)

# Convert the Spark DataFrame to a Pandas DataFrame
bookings_df = spark_df.toPandas()

# Display the Pandas DataFrame
bookings_df.head()

# COMMAND ----------

# Separate the feature matrix (X) and the target variable (y) from the original DataFrame

# Create the feature matrix 'X' by dropping the target column 'is_canceled'
# This matrix contains all the independent variables that will be used to predict the target.
# Each row represents a booking, and each column represents a feature such as 'lead_time', 'country', 'market_segment', etc.
# The 'is_canceled' column is excluded because it represents the dependent variable we want to predict.
X = bookings_df.drop(columns=['is_canceled'])

# Create the target variable 'y' by selecting the 'is_canceled' column from the original DataFrame
# This target variable indicates whether a booking was canceled (1) or not (0).
# 'y' will be used as the dependent variable during model training and evaluation.
y = bookings_df['is_canceled']

# COMMAND ----------

# Split the data into training and testing sets using an 80/20 split ratio

# The train_test_split function from scikit-learn is used to randomly split the dataset into two subsets:
# - The training set: This subset will be used to train the machine learning model.
# - The testing set: This subset will be used to evaluate the model's performance on unseen data.

# Arguments:
# - 'X' and 'y' are the feature matrix and target variable, respectively.
# - test_size=0.2 specifies that 20% of the data will be set aside for testing, while the remaining 80% will be used for training.
# - random_state=42 ensures that the split is reproducible. By setting a seed value (42), the function will produce the same split every time it is run, which is useful for consistency in model evaluation and comparison.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the record counts
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

# Step 1: Initialize the StandardScaler object
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
# It transforms the data to have a mean of 0 and a standard deviation of 1, which is
# important for many machine learning models that are sensitive to the scale of input features.
scaler = StandardScaler()

# Step 2: Fit the scaler on the training data and then transform it
# The fit_transform() method first computes the mean and standard deviation of X_train,
# and then transforms X_train by subtracting the mean and dividing by the standard deviation.
# This ensures that X_train is now standardized with mean 0 and variance 1.
X_train_scaled = scaler.fit_transform(X_train)

# Step 3: Use the same scaler to transform the test data
# The transform() method uses the mean and standard deviation computed from the training data
# to standardize X_test. This ensures that the test data is scaled consistently with the training data.
# Note: Do not use fit_transform() on test data, as it would compute different scaling parameters,
# leading to inconsistent scaling and possibly biased model evaluation.
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# def create_tf_serving_json(data):
#     return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

# def score_model(dataset):
#     url = 'https://adb-1376134742576436.16.azuredatabricks.net/serving-endpoints/cancel_prediction/invocations'
#     headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
#     ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
#     data_json = json.dumps(ds_dict, allow_nan=True)
#     response = requests.request(method='POST', headers=headers, url=url, data=data_json)
#     if response.status_code != 200:
#         raise Exception(f'Request failed with status {response.status_code}, {response.text}')
#     return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC #Code from the Inferencing Engine

# COMMAND ----------

# Function to create a JSON structure that can be sent to a TensorFlow Serving endpoint
def create_tf_serving_json(data):
    # If 'data' is a dictionary, create a dictionary with the key 'inputs' and a list of values from the input data
    # This is useful for handling TensorFlow models expecting a specific 'inputs' key
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

# Function to send data to a deployed model endpoint and get the prediction
def score_model(dataset):
    # URL of the Databricks serving endpoint for the model
    url = 'https://adb-1376134742576436.16.azuredatabricks.net/serving-endpoints/cancel_prediction/invocations'
    
    # Set up the headers, including an Authorization header with the Databricks token stored in environment variables
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    
    # Prepare the dataset in the correct format depending on whether it's a Pandas DataFrame or not
    # If it's a DataFrame, we use the 'dataframe_split' orientation for efficient transmission of data
    # Otherwise, we convert the data to a format compatible with TensorFlow Serving
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    
    # Convert the dataset dictionary to a JSON string, allowing NaN values to be included in the JSON (common in Pandas data)
    data_json = json.dumps(ds_dict, allow_nan=True)
    
    # Send a POST request to the model endpoint with the prepared JSON data and headers
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    
    # Check if the request was successful (status code 200), otherwise raise an exception with the error details
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
    # Return the JSON response from the model, which contains the prediction(s)
    return response.json()


# COMMAND ----------

# MAGIC %md
# MAGIC #Invoke the Scoring function

# COMMAND ----------

# Define a pipeline with a logistic regression model
# The pipeline consists of a single step: 'classifier', which is the LogisticRegression model.
# The LogisticRegression model is configured with:
# - max_iter=10000: This increases the maximum number of iterations the model will take to converge.
# - class_weight='balanced': This automatically adjusts the class weights to handle class imbalance in the dataset.
pipeline = Pipeline([
    ('classifier', LogisticRegression(max_iter=10000, class_weight='balanced'))
])

# Assume 'X_train_scaled' and 'y_train' are preprocessed training data and labels respectively
# Fit the pipeline (train the model) using the scaled training data
pipeline.fit(X_train_scaled, y_train)

# Specify how many predictions we want to compare (the number of samples from the test set)
num_predictions = 25

# Get predictions from the model served in production using the score_model function
# This function sends a request to the deployed model and returns predictions
# We're sending the first 'num_predictions' samples from 'X_test_scaled' to the served model
served_predictions = score_model(X_test_scaled[:num_predictions])

# Get predictions from the locally trained pipeline model on the same test data
# Here, the pipeline's 'predict()' function is used to get predictions from the logistic regression model
model_evaluations = pipeline.predict(X_test_scaled[:num_predictions])

# Convert the served model predictions to a Pandas Series
# Assuming 'served_predictions' is a dictionary with a key 'predictions' containing the prediction results
# This series is named 'Served Model Prediction' for easier comparison
served_predictions_series = pd.Series(served_predictions['predictions'], name="Served Model Prediction")

# Convert the predictions from the locally trained model (pipeline) into a Pandas Series
# This series is named 'Model Prediction' to distinguish it from the served model's predictions
model_evaluations_series = pd.Series(model_evaluations, name="Model Prediction")

# Create a DataFrame that compares the predictions from the served model and the local model
# The DataFrame has two columns: 'Model Prediction' (local model) and 'Served Model Prediction' (served model)
pred_df = pd.DataFrame({
    'Model Prediction': model_evaluations_series,
    'Served Model Prediction': served_predictions_series
})

# Display the DataFrame with both predictions for comparison
# This will print the DataFrame with the predictions side-by-side for each of the test samples
print(pred_df)


# COMMAND ----------

# MAGIC %md
# MAGIC #Verify that both columns have the same value

# COMMAND ----------

# Check if all values match between the two columns
if (pred_df['Model Prediction'] == pred_df['Served Model Prediction']).all():
    print("The columns have the same values.")
else:
    print("The columns have different values.")
