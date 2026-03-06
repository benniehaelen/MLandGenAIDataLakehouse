# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="display: flex; align-items: flex-start; gap: 2rem; margin-bottom: 1rem;">
# MAGIC   <img
# MAGIC     src="https://i.imgur.com/ITL8dZE.jpeg"
# MAGIC     style="width: 200px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);"
# MAGIC     alt="Book Cover"/>
# MAGIC   <div>
# MAGIC     <div style="margin-bottom: 0.5rem;">
# MAGIC       <img src="https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg" style="height: 24px; vertical-align: middle;"/>
# MAGIC       <span style="font-size: 1.5rem; font-weight: bold; vertical-align: middle;">&nbsp;ML and Generative AI in the Data Lakehouse</span>
# MAGIC     </div>
# MAGIC     <p style="margin: 0.3rem 0;"><b>Name:</b> chapter 03-02-Feature Engineering</p>
# MAGIC     <p style="margin: 0.3rem 0;"><b>Author:</b> Bennie Haelen</p>
# MAGIC     <p style="margin: 0.3rem 0;"><b>Date:</b> 1-27-2026</p>
# MAGIC     <p style="margin: 0.5rem 0;"><b>Purpose:</b> This notebook performs the feature engineering for chapter 4 of the book: <i>Machine Learning Use Case with MLflow</i></p>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <p style="margin: 1rem 0 0.5rem 0; font-size: 1.2rem;"><b>Table of Contents</b></p>
# MAGIC
# MAGIC <pre style="margin-left: 1rem; font-family: monospace; font-size: 0.9rem;">
# MAGIC   1. Read the Hotel Bookings CSV File and gather basic info
# MAGIC   2. Handle Missing Data
# MAGIC         2.1 Question: From what country originate mosts of the guests?
# MAGIC         2.2 Question: How much do guests pay per room per night?
# MAGIC         2.3 Question: How does the price vary per night over the year?
# MAGIC         2.3 Question: Which are the busy months?
# MAGIC         2.4 Question: How long do people stay at the hotels?
# MAGIC   3. Additional Data Cleaning
# MAGIC         3.1 The cancelation rate of city hotels is higher than resort hotels.
# MAGIC         3.2 The earlier the booking made, higher the chances of cancellation.
# MAGIC         3.3 Bookings for longer durations have lower cancellations
# MAGIC         3.4 A repeated guest is less likely to cancel current booking.
# MAGIC         3.5 Higher previous cancellations lead to cancellation of current bookings.
# MAGIC         3.6 If room assigned is not the reserved room type, customer might cancel.
# MAGIC         3.7 If # of booking changes made is high, chance of cancellation is low.
# MAGIC         3.8 Refundable bookings or those without deposit have higher cancellations.
# MAGIC         3.9 If the # of days in waiting list is high, cancelations are higher
# MAGIC   4. Encoding Categorical Features
# MAGIC
# MAGIC   5. Correlation Analysis
# MAGIC
# MAGIC   6. Create the Feature Table in Unity Catalog
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Perform the required imports
# MAGIC The primary libraries that we use are:
# MAGIC - NumPy for analytic arrays
# MAGIC - Pandas for DataFrames
# MAGIC - MatplotLib for generating plots
# MAGIC - Scikit-learn for encoding
# MAGIC - Databricks Feature Engineering for Unity Catalog integration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# Unity Catalog Feature Engineering imports
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import monotonically_increasing_id, col

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Set Basic Options for Pandas and MatplotLib

# COMMAND ----------

# This option will ensure that we always display all columns of our dataset
pd.set_option("display.max_columns", None)

# Make sure to generate inline plots
%matplotlib inline

# Set the plot style
plt.style.use('fivethirtyeight')

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the Hotel Bookings CSV File and gather basic info
# MAGIC This dataset was downloaded from the [Kaggle Web Site](https://www.kaggle.com/). 
# MAGIC
# MAGIC This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.

# COMMAND ----------

# Load the dataset
df = pd.read_csv('/dbfs/FileStore/datasets/hotel_bookings.csv')
                 
# Display the number of rows and columns
dataset_shape = df.shape
print(f'Dataset has {dataset_shape[0]:,} rows and {dataset_shape[1]} columns.')

# COMMAND ----------

# Display the top lines
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display all the columns with their data types and null counts

# COMMAND ----------

# Get an overview of data types and non-null counts
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display the key statistics for each column

# COMMAND ----------

# Get an overview of the data types, summary statistics and non-null counts
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Handle Missing Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Missing Values

# COMMAND ----------

# Find the number of missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / len(df)) * 100

# Create a dataframe to display missing values and percentages
missing_data = pd.DataFrame({'Total Missing': missing_values, 'Percent Missing': missing_percentage})
missing_data = missing_data[missing_data['Total Missing'] > 0]
missing_data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create a Bar Chart with the missing values

# COMMAND ----------

# Create the plot
plt.figure(figsize=(10, 3))
bars = plt.barh(missing_data.index, missing_data['Percent Missing'])
plt.xlabel('Percentage of Missing Values')
plt.title('Missing Data by Column')

# Reverse the order of columns to match descending percentage
plt.gca().invert_yaxis()  

# Annotate the bars with the total missing values
for bar, total in zip(bars, missing_data['Total Missing']):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{total:,}', va='center', ha='left')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We see that by far, we have the most missing values in company (94%), agent (14%), country (<1%) and a few in the children column.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove the 'company' column
# MAGIC Since we have 94% missing values in the company column, we decide to remove it from the DataFrame entirely.

# COMMAND ----------

# Removing the "company" feature
# axis=1 indicates we want to drop a column (not a row)
# inplace=True applies the operation to the original dataframe
df.drop("company", inplace=True, axis=1)

# Display the number of rows and columns
dataset_shape = df.shape
print(f'Dataset has {dataset_shape[0]:,} rows and {dataset_shape[1]} columns.')

# COMMAND ----------

# Double check that the company column was removed
column = 'company'
if {column}.issubset(df.columns) is False:
    print(f"The column: '{column}' has been successfully removed from the DataFrame.")
else:
    print(f"The column: '{column}' is still in the DataFrame, please check your work!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle the 'children' column (4 missing values)
# MAGIC Since we only have 4 missing values, we'll substitute with the most common value.

# COMMAND ----------

# Get the unique values in the children feature
df.children.value_counts()

# COMMAND ----------

# Most guests had zero children - substitute nulls with 0
df.children = df.children.fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle the 'country' column (488 missing values)

# COMMAND ----------

# Let's look at which country is most popular
df.country.value_counts().sort_values(ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Portugal (PRT) is the most common country, which makes sense as the hotels are located in Portugal.

# COMMAND ----------

# Fill in the missing values with 'PRT' (Portugal)
df['country'] = df['country'].fillna('PRT')

# Verify no more missing values
missing_country = df['country'].isnull().sum()
print(f"Number of missing values in the country column: {missing_country}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove rows where adults, babies, and children are all zero
# MAGIC These rows represent invalid bookings.

# COMMAND ----------

# Setup the filter for invalid rows
filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)

# Display count of invalid rows
print(f"Number of rows with zero guests: {filter.sum()}")

# Filter out these rows
df = df[~filter]
print(f"We now have {df.shape[0]:,} rows and {df.shape[1]} columns in our DataFrame.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle the 'agent' column (16,340 missing values)
# MAGIC Per Kaggle documentation, NULL means the booking was not made through an agent.

# COMMAND ----------

# How many unique agencies do we have?
print(f"Number of unique agents: {df.agent.nunique()}")

# Fill in zeros for null agents (meaning no agent was used)
df['agent'].fillna(0, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify all missing values are handled

# COMMAND ----------

# Find remaining missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_data = pd.DataFrame({'Total Missing': missing_values})
missing_data = missing_data[missing_data['Total Missing'] > 0]

if len(missing_data) == 0:
    print("All missing values have been handled successfully!")
else:
    print("Remaining missing values:")
    display(missing_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Additional Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove Duplicate Rows

# COMMAND ----------

# Count duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count:,}")

# COMMAND ----------

# Drop the duplicated rows, keeping the first occurrence
df.drop_duplicates(inplace=True, keep="first")
df.reset_index(drop=True, inplace=True)

# Verify duplicates removed
print(f"Duplicates remaining: {df.duplicated().sum()}")
print(f"We now have {df.shape[0]:,} rows and {df.shape[1]} columns.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop the 'reservation_status_date' column
# MAGIC This column has limited business value for our prediction model.

# COMMAND ----------

df.drop(columns=['reservation_status_date'], inplace=True)
print(f"Columns remaining: {df.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove rows with zero adults
# MAGIC These represent data quality issues.

# COMMAND ----------

# Count rows with zero adults
zero_adults = (df.adults == 0).sum()
print(f"Rows with zero adults: {zero_adults}")

# Drop these rows
df.drop(df[df["adults"] == 0].index, inplace=True)

# Verify
print(f"Rows with zero adults after cleanup: {(df.adults == 0).sum()}")
print(f"Final row count: {df.shape[0]:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Encoding Categorical Features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Identify Categorical Columns

# COMMAND ----------

# List all categorical features (object type)
categorical_columns = [column for column in df.columns if df[column].dtype == "object"]
print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

# COMMAND ----------

# Create a categorical dataframe
categorical_df = df[categorical_columns].copy()
categorical_df.reset_index(drop=True, inplace=True)

# Display unique values for each categorical column
for column in categorical_df.columns:
    print(f"Unique values in {column}: {categorical_df[column].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encode the Month Feature

# COMMAND ----------

# Map arrival_date_month to month numbers
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
categorical_df.loc[:, "arrival_date_month"] = categorical_df["arrival_date_month"].map(month_mapping)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label Encode High-Cardinality Columns

# COMMAND ----------

# Create an instance of the LabelEncoder
label_encoder = LabelEncoder()

# Label encode the 'country' column (177 unique values - too many for one-hot)
categorical_df.loc[:, "country"] = label_encoder.fit_transform(categorical_df["country"])

# Label encode the 'hotel' column
categorical_df.loc[:, "hotel"] = label_encoder.fit_transform(categorical_df["hotel"])

categorical_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-Hot Encode Remaining Categorical Columns

# COMMAND ----------

# Use one-hot encoding for columns with lower cardinality
one_hot_columns = ["meal", "market_segment", "distribution_channel", 
                   "reserved_room_type", "assigned_room_type", 
                   "deposit_type", "customer_type", "reservation_status"]

categorical_df = pd.get_dummies(data=categorical_df, columns=one_hot_columns)
categorical_df.reset_index(drop=True, inplace=True)

print(f"Categorical dataframe shape after encoding: {categorical_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Numerical DataFrame

# COMMAND ----------

# Drop all categorical columns to get numerical columns
numerical_df = df.drop(columns=categorical_columns, axis=1)
numerical_df.reset_index(drop=True, inplace=True)

print(f"Numerical dataframe shape: {numerical_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Concatenate Numerical and Categorical DataFrames

# COMMAND ----------

# Concatenate both dataframes along the column axis
final_df = pd.concat([numerical_df, categorical_df], axis=1)
final_df.reset_index(drop=True, inplace=True)

print(f"Combined dataframe shape: {final_df.shape}")
final_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Correlation Analysis
# MAGIC Remove highly correlated features to reduce redundancy and multicollinearity.

# COMMAND ----------

# Create the correlation matrix
correlation_matrix = final_df.corr().abs()

# Set the threshold for high correlation
threshold = 0.85

# COMMAND ----------

# MAGIC %md
# MAGIC ## Identify Highly Correlated Feature Pairs
# MAGIC We extract the upper triangle of the correlation matrix to identify pairs of features with correlation above our threshold.

# COMMAND ----------

# Create upper triangle mask
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find columns to drop (one from each highly correlated pair)
to_drop = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            column_name = correlation_matrix.columns[i]
            to_drop.append(column_name)

to_drop = list(set(to_drop))
print(f"Columns to drop due to high correlation: {to_drop}")

# COMMAND ----------

# Drop the highly correlated columns
final_df.drop(columns=to_drop, inplace=True)
final_df.reset_index(drop=True, inplace=True)

print(f"Final dataframe shape after removing correlated features: {final_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Feature Table in Unity Catalog
# MAGIC
# MAGIC Unity Catalog Feature Engineering provides centralized governance, lineage tracking, and discoverability for ML features. We'll create a feature table that can be used across multiple ML projects.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Catalog and Schema

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the catalog if it doesn't exist
# MAGIC CREATE CATALOG IF NOT EXISTS book_ai_ml_lakehouse;
# MAGIC
# MAGIC -- Create the schema for feature tables
# MAGIC CREATE SCHEMA IF NOT EXISTS book_ai_ml_lakehouse.feature_store;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the DataFrame for Feature Engineering

# COMMAND ----------

# Convert uint8 columns to int32 for Spark compatibility
final_df = final_df.astype({col: 'int32' for col in final_df.select_dtypes('uint8').columns})

# Clean column names (remove spaces and special characters)
final_df.columns = [col.replace(" ", "_").replace("/", "_") for col in final_df.columns]

# Add a unique booking_id as the primary key for the feature table
final_df['booking_id'] = range(1, len(final_df) + 1)

# Reorder columns to put booking_id first
cols = ['booking_id'] + [col for col in final_df.columns if col != 'booking_id']
final_df = final_df[cols]

print(f"Final dataframe shape: {final_df.shape}")
print(f"Primary key column added: 'booking_id'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Spark DataFrame

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(final_df)

# Display schema
spark_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Feature Table using Feature Engineering Client
# MAGIC
# MAGIC The Feature Engineering Client provides methods to create, write, and manage feature tables in Unity Catalog.

# COMMAND ----------

# Initialize the Feature Engineering Client
fe = FeatureEngineeringClient()

# Define the feature table name in Unity Catalog
feature_table_name = "book_ai_ml_lakehouse.feature_store.hotel_booking_features"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create or Replace the Feature Table

# COMMAND ----------

# Check if table exists and drop it if needed (for idempotency)
try:
    spark.sql(f"DROP TABLE IF EXISTS {feature_table_name}")
    print(f"Dropped existing table: {feature_table_name}")
except Exception as e:
    print(f"Table does not exist or could not be dropped: {e}")

# COMMAND ----------

# Create the feature table in Unity Catalog
# The primary_keys parameter specifies which column(s) uniquely identify each row
fe.create_table(
    name=feature_table_name,
    primary_keys=["booking_id"],
    df=spark_df,
    description="Hotel booking features for cancellation prediction model. "
                "Contains encoded categorical variables, numerical features, "
                "and correlation-filtered attributes derived from the Kaggle hotel bookings dataset."
)

print(f"Feature table created: {feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the Feature Table

# COMMAND ----------

# Read back the feature table to verify
feature_df = spark.table(feature_table_name)
print(f"Feature table row count: {feature_df.count():,}")
print(f"Feature table column count: {len(feature_df.columns)}")

# Display sample data
display(feature_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Feature Table Metadata

# COMMAND ----------

# Get feature table metadata
feature_table_info = fe.get_table(name=feature_table_name)
print(f"Feature Table Name: {feature_table_info.name}")
print(f"Primary Keys: {feature_table_info.primary_keys}")
print(f"Description: {feature_table_info.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Also Save as Delta File for Backward Compatibility
# MAGIC We maintain the Delta file output for compatibility with the model training notebook.

# COMMAND ----------

# Save as Delta file (without booking_id for training compatibility)
training_df = spark_df.drop("booking_id")

delta_file = "dbfs:/FileStore/datasets/hotel_bookings.delta"
training_df.write.format("delta").mode("overwrite").save(delta_file)

print(f"Delta file saved to: {delta_file}")
print(f"Training data shape: {training_df.count():,} rows x {len(training_df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary
# MAGIC
# MAGIC This notebook performed the following feature engineering steps:
# MAGIC
# MAGIC 1. **Data Loading**: Loaded 119,390 hotel booking records with 32 columns
# MAGIC 2. **Missing Value Handling**: 
# MAGIC    - Removed 'company' column (94% missing)
# MAGIC    - Filled 'children' with 0 (most common value)
# MAGIC    - Filled 'country' with 'PRT' (most common)
# MAGIC    - Filled 'agent' with 0 (no agent used)
# MAGIC 3. **Data Cleaning**:
# MAGIC    - Removed invalid rows (zero guests)
# MAGIC    - Removed duplicate rows
# MAGIC    - Dropped low-value columns
# MAGIC 4. **Feature Encoding**:
# MAGIC    - Label encoded high-cardinality columns (country, hotel)
# MAGIC    - One-hot encoded categorical columns (meal, market_segment, etc.)
# MAGIC 5. **Correlation Analysis**: Removed features with correlation > 0.85
# MAGIC 6. **Feature Table Creation**: 
# MAGIC    - Created Unity Catalog feature table with governance and lineage
# MAGIC    - Saved Delta file for model training compatibility
# MAGIC
# MAGIC The final feature set is ready for model training in the next notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC # End of Feature Engineering
