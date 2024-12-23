# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px; width:30%; height: auto;" src="https://i.imgur.com/FWzhbhX.jpeg"   />   
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 03-04-Model Predictions
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-12-2024
# MAGIC
# MAGIC    Purpose:   This notebook performs predictions with the models built previously
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Read the hotel-booking delta file as a Pandas dataframe
# MAGIC         2 - Load the model from MLflow as a native Scikit-learn model
# MAGIC               2-1 Build the model URI
# MAGIC               2-2 Load the model
# MAGIC               2-3 Apply standard scaling to our X variables
# MAGIC               2-4 Visualize the standard deviation and mean
# MAGIC               2-5 Fill in default value for the 'agent' column
# MAGIC         3 - Load the model as a Spark UDF
# MAGIC               3-1 Create the Spark UDF from the Model URI
# MAGIC               3-1 Create a Spark Dataframe from our scaled X values
# MAGIC               3-2 Run the predictions with the Spark UDF
# MAGIC               3-3 Double-check the predictions
# MAGIC         4 - Load the model as a generic PyFunc model
# MAGIC               4-1 Load the model and run the prediction
# MAGIC               4-2 Double-check the predictions
# MAGIC

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

import mlflow.pyfunc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC #Set Basic Options for Pandas and MatplotLib

# COMMAND ----------

# This option will ensure that we always display all rows of
# our dataset
pd.set_option("display.max_columns", None)

# Make sure to generate inline plots
%matplotlib inline

# Set  the plot style
plt.style.use('fivethirtyeight')

# COMMAND ----------

# MAGIC %md
# MAGIC #Read the hotel_bookings Delta file produced by Feature Engineering

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

# MAGIC %md
# MAGIC #Prepare our Test and Training Data

# COMMAND ----------

# MAGIC %md
# MAGIC ##Define our feature matrix (X), and our target variable (y) 

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

# MAGIC %md
# MAGIC ##Split the data into Training and Test data with a 30% split

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

# MAGIC %md
# MAGIC #Load the Model as a native Scikit-learn model

# COMMAND ----------

# MAGIC %md
# MAGIC ##Build the Model URI

# COMMAND ----------

# The unique identifier for the MLflow run where the model was logged.
# This ID corresponds to the specific experiment run in MLflow.
run_id = "06ab7ac6a53c4431a0db34827220ddf6"  

# The name of the model as it was saved during the experiment run.
# This is the name you used when logging the model with MLflow.
model_name = "Logistic_Regression_Model"  

# The URI (Uniform Resource Identifier) for the model.
# This URI follows the format: "runs:/<run_id>/<model_name>".
# It is used by MLflow to locate and load the model from the specified run.
model_uri = f"runs:/{run_id}/{model_name}"

# Print the model URI to verify that it is correct.
# This will output the full model URI to the console.
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load the model from MLflow as a native Scikit-learn Model

# COMMAND ----------

# Load the model from the specified model URI in MLflow.
# The model_uri should point to the location where the model was saved during a previous run.
# This function loads the scikit-learn model that was logged using mlflow.sklearn.log_model().
model = mlflow.sklearn.load_model(model_uri=model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make a prediction with the model

# COMMAND ----------

# Use the loaded model to make predictions
y_pred = model.predict(X_test_scaled)  

# Count the number of 0s and 1s in the predictions
num_zeros = sum(1 for pred in y_pred if pred == 0)
num_ones = sum(1 for pred in y_pred if pred == 1)

# Print the results
print(f"Number of 0s: {num_zeros:,}")
print(f"Number of 1s: {num_ones:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Load the Model as a Spark UDF

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create the Spark UDF from the model URI 

# COMMAND ----------

# Create a Spark UDF (User Defined Function) from a model logged in MLflow using the pyfunc interface.
# 'pyfunc' is a generic format in MLflow that supports loading models from various machine learning frameworks.
# This allows the loaded model to be used in a Spark DataFrame as a UDF for making predictions.

# spark: The active Spark session in Databricks or your Spark environment.
# model_uri: The URI of the MLflow model, which points to the location where the model was saved (logged) in MLflow.
# env_manager: Specifies which environment manager to use for the UDF. In this case, 'virtualenv' is used.
# 'virtualenv' tells MLflow to manage the environment using Python's virtual environment system instead of conda.

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, env_manager='virtualenv')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a Spark Dataframe from our scaled X Test Values

# COMMAND ----------

# Create a Spark dataframe from our scaled X Test Values
X_test_sp = spark.createDataFrame(X_test_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Run the predictions with the Spark UDF

# COMMAND ----------

# Import the 'struct' function from PySpark, which is used to group multiple columns into a single structure.
# This is useful when passing multiple features (columns) as inputs to the UDF (User Defined Function).
from pyspark.sql.functions import struct

# Apply the UDF (pyfunc_udf) to the Spark DataFrame (X_test_sp) and create a new column 'prediction'.
# 'withColumn()' is a PySpark function that adds a new column or replaces an existing column in a DataFrame.

# 'pyfunc_udf' is the UDF (created earlier using mlflow.pyfunc.spark_udf) which loads the MLflow model and allows us to make predictions.

# 'struct(*(X_test_sp.columns))' combines all columns from the DataFrame (X_test_sp) into a single struct, 
# so that the UDF can process them together as model inputs. The '*' operator unpacks the list of column names.

predicted_df = X_test_sp.withColumn('prediction', pyfunc_udf(struct(*(X_test_sp.columns))))

# 'display()' is used in Databricks notebooks to visually display the contents of the DataFrame.
# In this case, it will show the DataFrame 'predicted_df', which includes the original columns from 'X_test_sp'
# and a new 'prediction' column that contains the predictions made by the model.
display(predicted_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Double-Check the predictions

# COMMAND ----------

from pyspark.sql.functions import struct, col, expr

# If 'prediction' is an array, extract the first element
# Adjust based on your model output if it's a scalar, this step can be skipped
new_predicted_df = predicted_df.withColumn('prediction', col('prediction')[0])

# Count the number of 0 and 1 predictions
count_zeros = new_predicted_df.filter(col('prediction') == 0).count()
count_ones = new_predicted_df.filter(col('prediction') == 1).count()

# Display the counts
print(f"Number of 0 predictions: {count_zeros:,}")
print(f"Number of 1 predictions: {count_ones:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Load the model as a generic PyFunc model

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load the Model

# COMMAND ----------

# Import the 'pyfunc' module from MLflow, which provides functionality for loading models
# in a generic Python function (PyFunc) format. This format allows models from various frameworks 
# (scikit-learn, TensorFlow, etc.) to be used with a consistent API.
import mlflow.pyfunc

# Load the model from MLflow using the 'load_model' function from the 'pyfunc' module.
# 'model_uri' specifies the location of the model in MLflow (which can be local or remote).
# The model is loaded as a PyFunc model, which provides a universal interface to make predictions
# regardless of the underlying machine learning framework.
model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Make predictions on the scaled test data, which is stored in 'X_test_scaled'.
# 'X_test_scaled' is expected to be a Pandas DataFrame or NumPy array that contains the input features.
# The 'predict()' method is used to generate predictions using the loaded PyFunc model.
y_pred = model.predict(X_test_scaled)

# At this point, 'y_pred' will contain the model's predictions for each row of 'X_test_scaled',
# which could be either class labels (for classification) or continuous values (for regression).

# COMMAND ----------

# MAGIC %md
# MAGIC ##Double-check the predictions

# COMMAND ----------

# Count the number of 0s and 1s in the predictions
num_zeros = sum(1 for pred in y_pred if pred == 0)
num_ones = sum(1 for pred in y_pred if pred == 1)

# Print the results
print(f"Number of 0s: {num_zeros:,}")
print(f"Number of 1s: {num_ones:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC #End of Notebook
