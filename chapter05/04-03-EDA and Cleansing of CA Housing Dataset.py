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
TABLE_NAME   = "ca_housing_prices"

# COMMAND ----------

# MAGIC %md
# MAGIC #Read our Delta table with the CA Housing Prices

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
# MAGIC #Start our EDA and Data Cleanup

# COMMAND ----------

# MAGIC %md
# MAGIC ##Get the Row- and Column Count

# COMMAND ----------

# How many rows and columns are in the dataset?
df_housing_prices.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ##Drop all rows with missing data

# COMMAND ----------

# Drop any rows in the pandas DataFrame that contain missing values (NaN)
# This ensures that the DataFrame only contains complete cases for analysis or modeling
df_housing_prices = df_housing_prices.dropna()

# Display the shape of the DataFrame after dropping missing values
# .shape returns a tuple (number of rows, number of columns), giving insight into 
# how many rows and columns remain after removing rows with missing data
print(f"There are {df_housing_prices.shape[0]:,} rows left after dropping rows with missing data")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Let's take a look at our dataset now

# COMMAND ----------

 # Generate descriptive statistics of the pandas DataFrame
# .describe() provides summary statistics for numeric columns, including:
# - count: number of non-null values
# - mean: average value
# - std: standard deviation
# - min: minimum value
# - 25%, 50%, 75%: quartiles (percentiles)
# - max: maximum value
df_housing_prices.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create plots to investigate median_house_value and housing_median_age

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the overall figure size and layout
# figsize=(14, 10) sets a larger figure size for better visibility
plt.figure(figsize=(14, 10))

# Plot a histogram for 'median_house_value'
# 2 rows, 1 column, 1st subplot
plt.subplot(2, 1, 1)

# Plotting the histogram with Seaborn's histplot function
sns.histplot(
    data=df_housing_prices,        # The DataFrame to use
    x="median_house_value",        # Column to plot
    bins=50,                       # Set number of bins to 50 for finer granularity
    kde=True,                      # Add a kernel density estimate for smoothness
    color='skyblue'                # Set a distinct color for the histogram
)

# Add a title, x-label, and y-label to the plot for better understanding
plt.title('Distribution of Median House Value', fontsize=16)
plt.xlabel('Median House Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Plot a histogram for 'housing_median_age'
# 2 rows, 1 column, 2nd subplot
plt.subplot(2, 1, 2)

# Plotting the histogram for 'housing_median_age' column
sns.histplot(
    data=df_housing_prices,        # The DataFrame to use
    x="housing_median_age",        # Column to plot
    bins=50,                       # Set number of bins to 50 for finer granularity
    kde=True,                      # Add a kernel density estimate for smoothness
    color='salmon'                 # Set a different color for the histogram
)

# Add a title, x-label, and y-label to the plot for better understanding
plt.title('Distribution of Housing Median Age', fontsize=16)
plt.xlabel('Housing Median Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Adjust layout to prevent overlapping elements and improve spacing between plots
plt.tight_layout()

# Display the plots
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ##Process the data outliers

# COMMAND ----------

# MAGIC %md
# MAGIC From our describe() function earlier we know that:
# MAGIC 1. The max median_house_value is 500,001
# MAGIC 2. The max housing_media_age is 52
# MAGIC
# MAGIC We see large peaks for both of those in our histogram. 
# MAGIC Let's go find out how many records we have for each, we will likely have to drop those records

# COMMAND ----------

# MAGIC %md
# MAGIC ###How many outliers do we have for median_house_value?

# COMMAND ----------

# Filter the DataFrame to select rows where 'median_house_value' is exactly 500001
# .loc[] is used to access rows based on a condition
# df_housing_prices['median_house_value'] == 500001 creates a boolean mask 
# that is True only for rows where 'median_house_value' equals 500001
filtered_rows = df_housing_prices.loc[df_housing_prices['median_house_value'] == 500001]

# .count() counts the number of non-null values in each column of the filtered DataFrame
# Since we are interested in the total number of matching rows, we typically look at one column's count
num_rows_500001 = filtered_rows.count()

# Display the count of rows in the filtered DataFrame for each column
print(num_rows_500001)

# COMMAND ----------

# MAGIC %md
# MAGIC ###You decided to drop these records

# COMMAND ----------

# Identify the rows in the DataFrame where 'median_house_value' is exactly 500001
# df_housing_prices['median_house_value'] == 500001 creates a boolean mask
# df_housing_prices[...].index retrieves the index positions of the matching rows
rows_to_drop = df_housing_prices[df_housing_prices['median_house_value'] == 500001].index

# Drop the identified rows from the DataFrame
# .drop() removes the specified index positions from the DataFrame
df_housing_prices = df_housing_prices.drop(rows_to_drop)

# The DataFrame is now updated to exclude all rows where 'median_house_value' was 500001

# COMMAND ----------

print(f"We now have {df_housing_prices.shape[0]:,} rows in the DataFrame left")

# COMMAND ----------

# MAGIC %md
# MAGIC ###How many outliers do we have for housing_median_age

# COMMAND ----------

# Filter the DataFrame to select rows where 'housing_median_age' is exactly 52
# .loc[] is used to access rows based on a condition
# df_housing_prices['housing_median_age'] == 52 creates a boolean mask 
# that is True only for rows where 'housing_median_age' equals 52
filtered_rows = df_housing_prices.loc[df_housing_prices['housing_median_age'] == 52]

# .count() counts the number of non-null values in each column of the filtered DataFrame
# It provides the count for each column individually, but not the total row count
num_rows_52 = filtered_rows.count()

# Display the count of rows in the filtered DataFrame for each column
print(num_rows_52)

# COMMAND ----------

# MAGIC %md
# MAGIC ###You decided to drop these records

# COMMAND ----------

# Identify the rows in the DataFrame where 'housing_median_age' is exactly 52
# df_housing_prices['housing_median_age'] == 52 creates a boolean mask
# df_housing_prices[...].index retrieves the index positions of the matching rows
rows_to_drop = df_housing_prices[df_housing_prices['housing_median_age'] == 522].index

# Drop the identified rows from the DataFrame
# .drop() removes the specified index positions from the DataFrame
df_housing_prices = df_housing_prices.drop(rows_to_drop)

# The DataFrame is now updated to exclude all rows where 'housing_median_age' was 51

# COMMAND ----------

print(f"We now have {df_housing_prices.shape[0]:,} rows left")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Investigate the Categorical Columns

# COMMAND ----------

# MAGIC %md
# MAGIC ###How many unique values do we have for ocean_proximity"?

# COMMAND ----------

# Retrieve the unique values from the 'ocean_proximity' column of the DataFrame
# .unique() returns an array of unique values present in the specified column
# This helps in identifying all distinct categories or labels in the 'ocean_proximity' column
unique_values = df_housing_prices['ocean_proximity'].unique()

# Display the unique values in the 'ocean_proximity' column
print(unique_values)

# COMMAND ----------

# MAGIC %md
# MAGIC ###We will do a 'one-hot encoding' for ocean_proximity

# COMMAND ----------

# Convert the 'ocean_proximity' column into one-hot encoded variables
# pd.get_dummies() is used to perform one-hot encoding, which converts categorical variables
# into separate binary columns (0 or 1) for each category in the original column
# 'columns=['ocean_proximity']' specifies that only the 'ocean_proximity' column should be encoded
df_housing_prices = pd.get_dummies(df_housing_prices, columns=['ocean_proximity'])

# The DataFrame is now updated with new columns, each representing a category from 'ocean_proximity'
# For example, if 'ocean_proximity' had values like 'NEAR OCEAN' and 'INLAND', 
# the DataFrame will have new columns: 'ocean_proximity_NEAR OCEAN', 'ocean_proximity_INLAND', etc.
# Each new column will have 1s and 0s indicating the presence of that category for each row

# COMMAND ----------

print(f"We now have {df_housing_prices.shape[0]:,} rows and {df_housing_prices.shape[1]} columns.")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Let's take a look at our new columns

# COMMAND ----------

df_housing_prices.sample(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##More Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a scatter plot of total_rooms against median_house_value 

# COMMAND ----------

import matplotlib.pyplot as plt

# Create a figure and axis with a specified size
# figsize=(12, 8) sets the figure size to 12 inches wide and 8 inches high
fig, ax = plt.subplots(figsize=(12, 8))

# Plot a scatter plot of 'total_rooms' vs. 'median_house_value'
# 'alpha=0.6' sets the transparency to 60%, making dense areas easier to see
# 'color='dodgerblue'' sets a distinct blue color for the scatter points
# 'edgecolor='k'' adds a black edge around each point for better contrast
# 'linewidth=0.5' sets the edge line width, enhancing the separation between points
ax.scatter(
    df_housing_prices['total_rooms'], 
    df_housing_prices['median_house_value'], 
    alpha=0.6,            
    color='dodgerblue',   
    edgecolor='k',        
    linewidth=0.5         
)

# Set the x-axis label to 'Total Rooms'
# 'fontsize=14' sets a larger font size for better readability
# 'labelpad=10' adds padding between the label and the axis for improved spacing
ax.set_xlabel('Total Rooms', fontsize=14, labelpad=10)

# Set the y-axis label to 'Median House Value'
# Similar settings for font size and label padding as the x-axis
ax.set_ylabel('Median House Value', fontsize=14, labelpad=10)

# Add a title to the scatter plot for context
# 'fontsize=16' sets a larger font size for the title
# 'pad=15' adds padding above the title for better spacing
ax.set_title('Scatter Plot of Total Rooms vs. Median House Value', fontsize=16, pad=15)

# Add grid lines to the plot for better visualization
# 'linestyle="--"' sets a dashed line style for the grid lines
# 'alpha=0.7' sets the transparency of the grid lines for a softer appearance
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust the layout to prevent overlapping of labels and elements
# tight_layout() ensures proper spacing between plot elements
plt.tight_layout()

# Display the scatter plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC We notice a definitive positive trend in the above scatter plot

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a scatterplot of median age against house value

# COMMAND ----------

import matplotlib.pyplot as plt

# Create a figure and axis with a specified size
# figsize=(12, 8) sets the figure size to 12 inches wide and 8 inches high
fig, ax = plt.subplots(figsize=(12, 8))

# Plot a scatter plot of 'housing_median_age' vs. 'median_house_value'
# 'alpha=0.6' sets the transparency to 60%, making dense areas easier to see
# 'color="seagreen"' sets a distinct green color for the scatter points
# 'edgecolor="black"' adds a black edge around each point for better contrast
# 'linewidth=0.5' sets the edge line width, enhancing the separation between points
ax.scatter(
    df_housing_prices['housing_median_age'], 
    df_housing_prices['median_house_value'], 
    alpha=0.6, 
    color='seagreen', 
    edgecolor='black',
    linewidth=0.5
)

# Set the x-axis label to 'Housing Median Age'
# 'fontsize=14' sets a larger font size for better readability
# 'labelpad=10' adds padding between the label and the axis for improved spacing
ax.set_xlabel('Housing Median Age', fontsize=14, labelpad=10)

# Set the y-axis label to 'Median House Value'
# Similar settings for font size and label padding as the x-axis
ax.set_ylabel('Median House Value', fontsize=14, labelpad=10)

# Add a title to the scatter plot for context
# 'fontsize=16' sets a larger font size for the title
# 'pad=15' adds padding above the title for better spacing
ax.set_title('Scatter Plot of Housing Median Age vs. Median House Value', fontsize=16, pad=15)

# Add grid lines to the plot for better visualization
# 'linestyle="--"' sets a dashed line style for the grid lines
# 'alpha=0.7' sets the transparency of the grid lines for a softer appearance
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust the layout to prevent overlapping of labels and elements
# tight_layout() ensures proper spacing between plot elements
plt.tight_layout()

# Display the scatter plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC In the above diagram, we see no real trend

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a scatterplot of median income against median house value

# COMMAND ----------

import matplotlib.pyplot as plt

# Create a figure and axis with a specified size
# figsize=(12, 8) sets the figure size to 12 inches wide and 8 inches high for better visibility
fig, ax = plt.subplots(figsize=(12, 8))

# Plot a scatter plot of 'median_income' vs. 'median_house_value'
# 'alpha=0.6' sets the transparency to 60%, making dense areas easier to see
# 'color="darkorange"' sets a distinct orange color for the scatter points
# 'edgecolor="black"' adds a black edge around each point for better contrast
# 'linewidth=0.5' sets the edge line width, enhancing separation between points
ax.scatter(
    df_housing_prices['median_income'], 
    df_housing_prices['median_house_value'], 
    alpha=0.6, 
    color='darkorange', 
    edgecolor='black',
    linewidth=0.5
)

# Set the x-axis label to 'Median Income'
# 'fontsize=14' sets a larger font size for better readability
# 'labelpad=10' adds padding between the label and the axis for improved spacing
ax.set_xlabel('Median Income', fontsize=14, labelpad=10)

# Set the y-axis label to 'Median House Value'
# 'fontsize=14' sets a larger font size for the y-axis label
# 'labelpad=10' adds padding between the label and the axis
ax.set_ylabel('Median House Value', fontsize=14, labelpad=10)

# Add a title to the scatter plot for context
# 'fontsize=16' sets a larger font size for the title
# 'pad=15' adds padding above the title for better spacing
ax.set_title('Scatter Plot of Median Income vs. Median House Value', fontsize=16, pad=15)

# Add grid lines to the plot for better visualization
# 'linestyle="--"' sets a dashed line style for the grid lines
# 'alpha=0.7' sets the transparency of the grid lines for a softer appearance
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust the layout to prevent overlapping of labels and elements
# tight_layout() ensures proper spacing between plot elements
plt.tight_layout()

# Display the scatter plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC You can see a definite positive relationship between the two variables

# COMMAND ----------

# MAGIC %md
# MAGIC ###Function to create a correlation matrix from a Spark Dataframe.
# MAGIC We had to create this function because the pandas corr() function ran out out stack space, so we switched to a Spark Dataframe

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import pandas as pd

def create_correlation_matrix(df):
    # Step 1: Identify numeric columns
    # Create a list of column names that have numeric data types ('int', 'double', 'float')
    # These are the columns we want to include in the correlation matrix
    numeric_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double', 'float']]
    
    # Step 2: Select only numeric columns from the DataFrame
    df_numeric = df.select(numeric_cols)

    # Step 3: Assemble the numeric columns into a single vector column
    # VectorAssembler is used to combine multiple numeric columns into one vector column
    # 'inputCols=numeric_cols' specifies which columns to assemble
    # 'outputCol="features"' creates a new column named 'features' that contains the assembled vectors
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df_vector = assembler.transform(df_numeric).select("features")

    # Step 4: Compute the correlation matrix
    # Correlation.corr() computes the Pearson correlation matrix for the 'features' vector column
    # 'pearson' is specified as the correlation method (default option)
    corr_matrix = Correlation.corr(df_vector, "features", "pearson").head()[0]

    # Step 5: Convert the resulting matrix to a dense matrix (NumPy array)
    # toArray() converts the Spark matrix to a NumPy array for easier handling
    corr_matrix = corr_matrix.toArray()

    # Step 6: Display the correlation matrix as a pandas DataFrame for better visualization
    # Convert the NumPy array to a pandas DataFrame with row and column labels
    # 'index=numeric_cols' and 'columns=numeric_cols' set the labels for the rows and columns
    corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)
    print(corr_df)  # Print the correlation matrix for verification

    # Return the pandas DataFrame containing the correlation matrix
    return corr_df


# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a Correlation Matrix for our dataset features

# COMMAND ----------

# Call the create_correlation_matrix function with a Spark DataFrame
# spark.createDataFrame(df_housing_prices) converts the pandas DataFrame (df_housing_prices)
# to a Spark DataFrame, which is required as input for the create_correlation_matrix function
corr_matrix = create_correlation_matrix(spark.createDataFrame(df_housing_prices))

# Display the correlation matrix
# corr_matrix contains the correlation matrix as a pandas DataFrame
# Printing or displaying it shows the pairwise correlation values between numeric columns
corr_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC ###Visualize the Correlation Matrix

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure size and layout
# figsize=(12, 10) increases the size of the plot to make it more readable
plt.figure(figsize=(12, 10))

# Create the heatmap using Seaborn
# 'data=corr_matrix' specifies the pandas DataFrame containing the correlation matrix
# 'annot=True' adds the correlation values inside the heatmap cells for clarity
# 'cmap="coolwarm"' sets a diverging color palette, with blue representing negative correlations,
# red representing positive correlations, and white representing no correlation
# 'vmin=-1' and 'vmax=1' ensure that the color scale ranges from -1 to 1
# 'linewidths=0.5' adds thin lines between cells for better separation
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    linewidths=0.5, 
    square=True,          # Make cells square-shaped for better proportion
    annot_kws={"size": 12}  # Set the font size for the annotation text
)

# Add a title to the heatmap
# 'fontsize=18' sets a larger font size for the title
plt.title('Correlation Heatmap for California Housing Prices', fontsize=18, pad=20)

# Adjust the layout to ensure proper spacing and prevent clipping
plt.tight_layout()

# Display the heatmap
plt.show()


# COMMAND ----------

df_housing_prices.sample(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #Write the Pandas Dataframe to a Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create the full table name

# COMMAND ----------

# The name of our Calfornia Housing Prices Features Delta Table
TABLE_NAME   = "ca_housing_prices_features"

# Create the full table name
full_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"

print(f"Full feature table name: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Convert our Pandas Dataframe to a Spark DataFrame

# COMMAND ----------

# Convert our Pandas DataFrame to a Spark Dataframe
df_spark = spark.createDataFrame(df_housing_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ##When we look at our columns, we see some invalid characters

# COMMAND ----------

# Print out the spark columns, we notice the "<" character is not allowed in column names
print(df_spark.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #Fix the column names to be valid Unity columns

# COMMAND ----------

# Replace or remove special characters in column names of the Spark DataFrame
# 'df_spark.columns' returns a list of the current column names in the DataFrame
# The list comprehension iterates over each column name ('col') and applies the following replacements:
# 1. 'col.replace('<', 'less_than_')' replaces the '<' character with 'less_than_'
#    (e.g., 'ocean_proximity_<1H OCEAN' becomes 'ocean_proximity_less_than_1H_OCEAN')
# 2. 'col.replace(' ', '_')' replaces spaces with underscores ('_')
#    (e.g., 'ocean_proximity_NEAR BAY' becomes 'ocean_proximity_NEAR_BAY')
# This transformation ensures that the column names are compliant with Delta Lake's naming rules
df_spark = df_spark.toDF(*[col.replace('<', 'less_than_').replace(' ', '_') for col in df_spark.columns])

# Display the new column names for verification
# 'print(df_spark.columns)' prints the updated column names, allowing you to check that special characters 
# have been replaced correctly and that column names now follow standard conventions
print(df_spark.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Write the DataFrame to the Features Delta Table

# COMMAND ----------

# Write the Spark DataFrame to a managed Delta table in the specified catalog
# 'df_spark' is the Spark DataFrame that contains the data you want to save as a Delta table
# 'format("delta")' specifies that the data should be written in Delta Lake format, which supports ACID transactions
# 'mode("overwrite")' indicates that if a table with the same name already exists, it will be replaced
# This mode is useful for updating or replacing the table with new data
# 'saveAsTable(full_table_name)' saves the DataFrame as a managed Delta table in the catalog
# 'full_table_name' is a string representing the fully qualified table name in the format 'catalog.schema.table_name'
df_spark.write.format("delta").mode("overwrite").saveAsTable(full_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #End of notebook
