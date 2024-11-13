# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px;" src="https://learning.oreilly.com/covers/urn:orm:book:9781098139711/400w/" />  
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 04-03-Regression with AutoML
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-24-2024
# MAGIC
# MAGIC    Purpose:   This notebook will perform EDA on the dataset, and will run AutoML using the Python API
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Make sure kaggle and kagglehub are installed
# MAGIC         2 - Use KaggleHub to download the Kaggle Dataset
# MAGIC         3 - Copy the local file to our DBFS datasets location
# MAGIC         4 - Read the source file from our dbfs location
# MAGIC         5 - Create our Catalog and Schema (if needed)
# MAGIC         6 - Save our Dataframe as a Delta Table in our Catalog
# MAGIC               
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Run our Constants notebook

# COMMAND ----------

# MAGIC %run "../common/Constants"

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn .preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import mlflow
from databricks import automl

# COMMAND ----------

# MAGIC %md
# MAGIC #Define our constants

# COMMAND ----------

# The name of our Calfornia Housing Prices Delta Table
TABLE_NAME   = "ca_housing_prices_features"

# COMMAND ----------

# MAGIC %md
# MAGIC #Read our Delta table with the CA Housing Prices Features

# COMMAND ----------

# Combine catalog, schema, and table names to create the fully qualified table path
# The format is 'catalog.schema.table', which ensures that the table is accessed 
# from the specified catalog and schema in Unity Catalog
full_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"

# Read the Delta table from Unity Catalog using the fully qualified table name
# The .toPandas() method converts the Spark DataFrame to a pandas DataFrame
# This is useful for further analysis or visualization in Python
df_housing_prices = spark.read.table(full_table_name).toPandas()

# Display a random sample of 5 rows from the pandas DataFrame
# This provides a quick look at the data for verification or exploration
df_housing_prices.sample(5)


# COMMAND ----------

# MAGIC %md
# MAGIC #Start the Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ##Perform a train/test split of the dataframe

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the pandas DataFrame into training and testing sets
# 'train_test_split()' is used to split the data into random train and test subsets
# 'df_housing_prices' is the original DataFrame to be split
# 'test_size=0.2' specifies that 20% of the data should go into the test set, while 80% goes into the training set
# 'random_state=42' ensures that the split is reproducible, so you get the same train-test split each time you run it
train_df, test_df = train_test_split(df_housing_prices, test_size=0.2, random_state=42)

# Display the first few rows of the training DataFrame
# 'train_df.head()' shows the first 5 rows of the training DataFrame, allowing you to verify the split
train_df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ##Investigate the shapes of the training and test datasets

# COMMAND ----------

# Display the rows and columns of both test and train dataframes
train_df.shape, test_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ##Convert our Training Pandas Dataframe to a Spark Dataframe

# COMMAND ----------

# Convert the Pandas Dataframe to a Spark Dataframe,
# since the AutoMLRegressor class expects a Spark Dataframe
train_df_spark = spark.createDataFrame(train_df)

# COMMAND ----------

# Display the columns of the Spark Dataframe
train_df_spark.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##Start the AutoML Regression 

# COMMAND ----------

# Run AutoML regression on the training Spark DataFrame
# 'automl.regress()' is an AutoML function that automatically finds the best regression model
# for the given dataset and target column
# 'train_df_spark' is the input Spark DataFrame containing the training data
# 'target_col='median_house_value'' specifies the target column to predict (dependent variable)
# 'timeout_minutes=10' sets a timeout of 10 minutes, after which the AutoML process will stop,
# even if it hasn't tried all possible models
summary = automl.regress(
    train_df_spark,
    target_col='median_house_value',
    timeout_minutes=10
)

# 'summary' will contain information about the best model found by AutoML, along with metrics,
# feature importance, and other details that can be used for evaluation and interpretation


# COMMAND ----------

# MAGIC %md
# MAGIC #Study the results of the regression and make predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ##Retrieve the best model's URI

# COMMAND ----------

# Retrieve the model path for the best model found by AutoML
# 'summary.best_trial' refers to the best trial (or run) identified during the AutoML process
# 'model_path' is the attribute that contains the location (URI) of the saved best model
model_uri = summary.best_trial.model_path

# Display the model URI
# 'model_uri' holds the path where the best model is stored, typically in a location like MLflow
# or a managed model registry, making it easy to load and use the model later for inference
model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create the Test Features Dataframe and the Target Series

# COMMAND ----------

# Create the test features DataFrame by dropping the 'median_house_value' column
# 'test_df.drop()' removes the specified column from the DataFrame
# 'axis=1' indicates that the operation is applied to columns (not rows)
# 'X_test' will contain all the feature columns from the test DataFrame, except for the target column
X_test = test_df.drop('median_house_value', axis=1)

# Create the test target Series by selecting the 'median_house_value' column
# 'y_test' will hold the true target values from the test DataFrame, 
# which will be used to evaluate the model's predictions
y_test = test_df['median_house_value']

# COMMAND ----------

X_test.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load the best model from MLflow in a Python function format

# COMMAND ----------

import mlflow.pyfunc

# Load the best model from the specified URI using MLflow
# 'mlflow.pyfunc.load_model()' loads a model saved in MLflow in a generic Python function format
# 'model_uri' is the path to the best model found during the AutoML process (extracted earlier)
# This model can now be used for inference (making predictions) on new data
model = mlflow.pyfunc.load_model(model_uri)

# Display the loaded model
# 'model' now contains the loaded model object, ready for use in prediction tasks or evaluation
model

# COMMAND ----------

# MAGIC %md
# MAGIC ##Use the loaded model to make predictions on the Test Feature

# COMMAND ----------

train_df_spark.columns

# COMMAND ----------

X_test.columns

# COMMAND ----------

X_test = X_test.rename(columns={
    'ocean_proximity_less_than_1H_OCEAN': 'ocean_proximity_<1H OCEAN',
    'ocean_proximity_NEAR_BAY': 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR_OCEAN': 'ocean_proximity_NEAR OCEAN'
})

# COMMAND ----------

# Use the loaded model to make predictions on the test features
# 'model.predict(X_test)' takes the test features (X_test) as input and generates predicted values
# for the target variable ('median_house_value') based on the model's learned patterns
predictions = model.predict(X_test)

# Add the predicted values to the original test DataFrame as a new column
# 'Median House Value Predicted' is the name of the new column that stores the model's predictions
# This allows for comparison between the actual and predicted values in the same DataFrame
test_df['Median House Value Predicted'] = predictions

# Display the first few rows of the test DataFrame, including the new prediction column
# 'test_df.head()' shows the first 5 rows, allowing you to verify the added predictions
test_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Extract predicted features & combine the actual and predicted values in a Dataframe

# COMMAND ----------

# Extract the predicted values from the test DataFrame
# 'y_pred' is set to the values from the 'Median House Value Predicted' column,
# which contains the predictions made by the model
y_pred = test_df['Median House Value Predicted']

# Create a new pandas DataFrame to compare predicted and actual values
# 'pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})' constructs a new DataFrame
# with two columns: 'Predicted' and 'Actual', allowing for side-by-side comparison
# 'y_pred' contains the model's predicted values, while 'y_test' contains the true target values
test = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})

# Reset the index of the new DataFrame
# This ensures that the DataFrame has a clean, sequential index starting from 0
test = test.reset_index()

# Drop the old index column that was added during the reset operation
# 'axis=1' specifies that the drop operation is applied to columns, not rows
test = test.drop('index', axis=1)                

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a plot comparing the actuals with the predictions

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Option 1: Smooth the lines using a rolling average
# 'window=10' sets the size of the rolling window, making it smoother
test['Actual_Smoothed'] = test['Actual'].rolling(window=10).mean()
test['Predicted_Smoothed'] = test['Predicted'].rolling(window=10).mean()

# Option 2: Sample the data to plot fewer observations (e.g., every 10th point)
# This helps to reduce clutter and make the trends more visible
test_sampled = test[::10]  # Takes every 10th row for visualization

# Set up the figure size for the plot
plt.figure(figsize=(16, 8))

# Plot the smoothed 'Actual' and 'Predicted' values from the test DataFrame
plt.plot(test_sampled['Actual_Smoothed'], label='Actual (Smoothed)', color='blue', linewidth=2, alpha=0.8)
plt.plot(test_sampled['Predicted_Smoothed'], label='Predicted (Smoothed)', color='red', linestyle='--', linewidth=2, alpha=0.8)

# Add a fill between the two lines to highlight differences
plt.fill_between(test_sampled.index, test_sampled['Actual_Smoothed'], test_sampled['Predicted_Smoothed'],
                 color='gray', alpha=0.2, label='Difference')

# Add a legend to indicate which line represents actual and predicted values
plt.legend(loc='upper right', fontsize=12)

# Add a title to the plot for better context
plt.title('Smoothed Actual vs. Predicted Median House Value', fontsize=18, pad=20)

# Add x-axis and y-axis labels for clarity
plt.xlabel('Observations (Sampled)', fontsize=14, labelpad=10)
plt.ylabel('Median House Value', fontsize=14, labelpad=10)

# Add grid lines for better visualization of the trend
plt.grid(True, linestyle='--', alpha=0.5)

# Adjust layout to ensure proper spacing and prevent clipping
plt.tight_layout()

# Display the plot
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a Joint Plot of actual versus predicted

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure size for the joint plot
# figsize=(16, 8) creates a large figure, providing a clear view of the joint distribution
plt.figure(figsize=(16, 8))

# Create a joint plot to compare 'Actual' and 'Predicted' values from the test DataFrame
# 'x='Actual'' sets the 'Actual' column on the x-axis
# 'y='Predicted'' sets the 'Predicted' column on the y-axis
# 'data=test' specifies the DataFrame to use for the plot
# 'kind='reg'' adds a regression line to visualize the linear relationship between actual and predicted values
joint_plot = sns.jointplot(
    x='Actual', 
    y='Predicted', 
    data=test, 
    kind='reg', 
    height=8,            # Set height of the plot for better scaling
    scatter_kws={'alpha': 0.6, 'color': 'purple'},  # Adjust scatter transparency and color
    line_kws={'color': 'red', 'linewidth': 2}       # Set color and width for the regression line
)

# Add a title to the joint plot for better context
# The title is added to the joint plot's main figure
joint_plot.fig.suptitle(
    'Actual vs. Predicted Median House Value (Joint Plot)', 
    fontsize=18, 
    y=1.05  # Adjust title position slightly above the plot
)

# Add labels for the x-axis and y-axis
# The xlabel and ylabel provide clarity on what each axis represents
joint_plot.set_axis_labels('Actual Median House Value', 'Predicted Median House Value', fontsize=14)

# Adjust layout to ensure proper spacing and prevent overlapping of elements
plt.tight_layout()

# Display the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #End of Notebook
