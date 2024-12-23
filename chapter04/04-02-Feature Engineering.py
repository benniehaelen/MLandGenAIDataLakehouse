# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px; width:30%; height: auto;" src="https://i.imgur.com/FWzhbhX.jpeg"   />    
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 03-02-Feature Engineering
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-09-2024
# MAGIC
# MAGIC    Purpose:   This notebook performs the feature engineering for the chapter 3 use case: <br/> Intro to ML on Databricks
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Read the hotel-booking.csv notebook and display key statistics
# MAGIC         2 - Create a list with all columns that have nulls
# MAGIC         3 - Perform Feature Engineering
# MAGIC               3-1 Remove the 'company' column
# MAGIC               3-2 Subsitute zero for nulls in the 'children' column
# MAGIC               3-3 Fill in a default value for the 'country' column
# MAGIC               3-4 Remove all rows with zero adults, children and babies
# MAGIC               3-5 Fill in default value for the 'agent' column
# MAGIC         4 - More Data Cleansing
# MAGIC               4-1 Remove duplicates
# MAGIC               4-2 Drop the 'reservation_status_date' column
# MAGIC         5 - Perform Logic Checks
# MAGIC               5-1 Create a fitting default value for the 'adults' column
# MAGIC         6 - Encoding Categorical Features
# MAGIC               6-1 Create a categorical data frame
# MAGIC               6-2 Label encode the 'country' column
# MAGIC               6-3 Label encode the 'hotel' column
# MAGIC               6-4 One-hot encode all other categorical columns
# MAGIC               6-5 Create a numerical dataframe
# MAGIC               6-6 Concatenate the categorical and numerical dataframes
# MAGIC         7 - Correlation Analysis
# MAGIC               7-1 Create the correlation matrix
# MAGIC               7-2 Create the upper triangle of the matrix
# MAGIC               7-3 Get the highly correlated features, identify drop candidates
# MAGIC               7-4 Drop the candidate columns
# MAGIC         8 - Save the final dataframe as a Delta file
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #####Perform the required imports
# MAGIC The primary libraries that we use are:
# MAGIC - NumPy for analytic arrays
# MAGIC - Pandas for DataFrames
# MAGIC - MatplotLib for generating plots
# MAGIC - Scikit-learn for all machine learning tasks

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

# MAGIC %md
# MAGIC #####Set Basic Options for Pandas and MatplotLib

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
# MAGIC # Read the Hotel Bookings CSV File and gather basic info
# MAGIC This dataset was downloaded from the [Kaggle Web Site](https://www.kaggle.com/). 
# MAGIC
# MAGIC This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.

# COMMAND ----------

# Load the dataset
df = pd.read_csv('/dbfs/FileStore/datasets/hotel_bookings.csv')
                 
# Display trhe number of rows and columns
dataset_shape = df.shape
print(f'Dataset has {dataset_shape[0]:,} rows and {dataset_shape[1]} columns.')

# COMMAND ----------

# Display the top lines
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Display all the columns with their data types and null counts

# COMMAND ----------

# Get an overview of data types and non-null counts
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Display the key statistics for each column

# COMMAND ----------

# Get an overview of the data types, summary statistics and non-null counts
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #Perform Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ##First, deal with any missing data

# COMMAND ----------

# MAGIC %md
# MAGIC ###Get an indea of the current status

# COMMAND ----------

# MAGIC %md
# MAGIC #####Create a list

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
# MAGIC #####Create a Bar Chart with the missing values

# COMMAND ----------

# Create the plot
plt.figure(figsize=(10, 3))
bars = plt.barh(missing_data.index, missing_data['Percent Missing'])
plt.xlabel('Totals of Missing Values')
plt.title('Missing Data Numbers by Column')

# Reverse the order of columns to match descending percentage
plt.gca().invert_yaxis()  

# Annotate the bars with the total missing values
for bar, total in zip(bars, missing_data['Total Missing']):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{total:,}', va='center', ha='left')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We see that by bar, we have the most missing values in company, agent, country and a few in the children column.
# MAGIC Let's start with the company column

# COMMAND ----------

# MAGIC %md
# MAGIC ###First, let's look at company column with many misssing values

# COMMAND ----------

# MAGIC %md
# MAGIC #####Since we have 94% miss of all company values, we decide to remove the column from the DataFrame

# COMMAND ----------

# Removing the "company" features, note that we need to 
# use the "axis=1" attribute to indicate that we want to
# drop a column. By default, axis is set to zero, which 
# implies that we want to remove a row.
# Notice also that we use the inplace=True parameter, which
# implies that we do not want to create a new dataframe, but
# instead apply the operation "in place" in the original
# dataframe
df.drop("company", inplace=True, axis=1)

# Display trhe number of rows and columns
dataset_shape = df.shape
print(f'Dataset has {dataset_shape[0]:,} rows and {dataset_shape[1]} columns.')

# COMMAND ----------

# MAGIC %md
# MAGIC Let's make sure that the `company` column is gone

# COMMAND ----------

# Double check that the company column did get removed
column = 'company'
if {column}.issubset(df.columns) is False:
    print(f"The column: '{column}' is not in the DataFrame.")
else:
    print(f"The column: '{column}' is still in the DataFrame, please check your work!")


# COMMAND ----------

# Let's take a look at the updated DataFrame
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Next, let's take a look at the 'children' column, which had four missing values
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #####Since we only have 4 missing values, it makes sense to use the most common value

# COMMAND ----------

# Get the unique values in the children features, sort them
# to find the most common value
df.children.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the guests had zero children, which makes sense when you think about it, most vacationers want some private time without the kids. <br/>
# MAGIC We will substitute the nulls with this value.

# COMMAND ----------

# MAGIC %md
# MAGIC #####Substitute the nulls with a zero valule

# COMMAND ----------

# Fill in the children feature with 0
df.children = df.children.fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Next, lets look at the 'country' column with 488 missing values

# COMMAND ----------

# MAGIC %md
# MAGIC #####What country occurs most frequently?
# MAGIC

# COMMAND ----------

# Let's look at which country is most popular
df.country.value_counts().sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #####We see that the most common country is Portugal, 
# MAGIC This makes sense because if you check the EDA notebook, you will see that both hotels are located in Portual. We will fill in the nulls with 'PRT' for Portugal

# COMMAND ----------

# We are going to fill in the missing values in the country column 
# with 'PRT' (Portugal)
df['country'] = df['country'].fillna('PRT')

# COMMAND ----------

# Let's make sure that we have no more missing values in the country column
missing_country = df['country'].isnull().sum()
print(f"Number of missing values in the country column: {missing_country}")

# COMMAND ----------

# Let's take another look at our DataFrame
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adults, babies and children cannot be zero at the same time, so it is OK to drop the rows 

# COMMAND ----------

# MAGIC %md
# MAGIC #####Setup the filter

# COMMAND ----------

# Setup our filter
filter = (df.children == 0 ) & (df.adults == 0) & (df.babies == 0)

# Display the rows that match our filter
df[filter]

# COMMAND ----------

# MAGIC %md
# MAGIC #####Let's filter out those rows

# COMMAND ----------

# Filter out the rows
df = df[~filter]

# Let's take another look at the shape of our data
print(f"We now have {df.shape[0]:,} rows and {df.shape[1]} columns in our DataFrame.")

# COMMAND ----------

# MAGIC %md
# MAGIC ###The next column to look at is 'agent' with 16,340 missing values

# COMMAND ----------

# Let's take a look at the distribution of the agent column
df.agent.value_counts().sort_values(ascending=False)

# COMMAND ----------

# How many unique agencies do we have?
df.agent.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC #####Study the Kaggle documentation
# MAGIC When we look at the documentation, we see that the dataset will have NULLs when the booking was not made through an agent. So, in this case it would be safe to substitute zero for a null

# COMMAND ----------

# MAGIC %md
# MAGIC #####Substitute zero for null agents

# COMMAND ----------

# Fill in zeros for null in agent
df['agent'].fillna(0, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###How do we look now for missing values?

# COMMAND ----------

# Find the number of missing values
missing_values  = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / len(df)) * 100

# Create a dataframe to display missing values and percentages
missing_data = pd.DataFrame({'Total Missing': missing_values, 'Percent Missing': missing_percentage})
missing_data = missing_data[missing_data['Total Missing'] > 0]
missing_data

# COMMAND ----------

# MAGIC %md
# MAGIC ####This completes the missing values analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ##Further Data Cleansing

# COMMAND ----------

# MAGIC %md
# MAGIC ### First, let's take a look at any duplicates

# COMMAND ----------

# MAGIC %md
# MAGIC #####How many duplicates do we have?

# COMMAND ----------

# Let's first see how many duplicates we have
df.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #####What are the duplicated rows?

# COMMAND ----------

# Let's take a look at these duplicated rows
df.loc[df.duplicated(), :]

# COMMAND ----------

# MAGIC %md
# MAGIC #####Let's drop all duplicated rows

# COMMAND ----------

# Drop the duplicated rows, we will keep the first occurrence
df.drop_duplicates(inplace=True, keep="first")
df.reset_index(drop=True, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #####Let's make sure that there are no more duplicates

# COMMAND ----------

# How many duplicates do we have now?
# Let's first see how many duplicates we have
df.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #####How many records do we have left?

# COMMAND ----------

# How many records do wse have left now?
print(f"We have {df.shape[0]:,} records and {df.shape[1]} columns left")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Drop the reservation_status_date column

# COMMAND ----------

# MAGIC %md
# MAGIC This column really has no business value, so it's safe to drop

# COMMAND ----------

# Drop the reservation status date
df.drop(columns=['reservation_status_date'], inplace=True)
# column_dtype = df['reservation_status_date'].dtype
# print(f"The data type of the 'reservation_status_date' column is: {column_dtype}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Perform Logic Checks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clearn up the 'adults' column

# COMMAND ----------

# MAGIC %md
# MAGIC #####How Many rows with zero adults did we have?

# COMMAND ----------

# How many rows with 0 adults did we have ?
(df.adults == 0).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #####We made the decision to drop these rows, since we have no good alternative

# COMMAND ----------

# Let's drop the rows with zero adults in place
df.drop(df[df["adults"] == 0].index, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #####Let's double check that we have no more rows with zero adults

# COMMAND ----------

# We should have zero records with zero adults now
(df.adults == 0).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #Encoding Categorical Features

# COMMAND ----------

# MAGIC %md
# MAGIC ##First, let's list all categorical features

# COMMAND ----------

# Creating a dataframe with just the categorical features from our original dataframe
# We do this by looking for columns that are of type "object"
categorical_columns = [column for column in df.columns if df[column].dtype == "object"]
categorical_columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a Categorical DataFrame

# COMMAND ----------

# Create a categorical dataframe
categorical_df = df[categorical_columns]
categorical_df.reset_index(drop=True, inplace=True)
categorical_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Let's take a look at all the unique values for each categorical column

# COMMAND ----------

# Display the unique values for each categorical column
for column in categorical_df.columns:
  print(f"Unique values in {column}: {categorical_df[column].unique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Map the Month Categorical Feature Map to ints

# COMMAND ----------

# Manually mapping arrival_date_month to a month number
categorical_df.loc[:, "arrival_date_month"] = \
    categorical_df["arrival_date_month"].map({"January": 1, "February": 2, "March": 3, 
                                              "April": 4, "May": 5, "June": 6, "July": 7,
                                              "August": 8, "September": 9, "October": 10,
                                              "November": 11, "December": 12})
categorical_df.reset_index(drop=True, inplace=True)
categorical_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Label encode the country column

# COMMAND ----------

# Create an instance of the LabelEncoder
label_encoder = LabelEncoder()

# Label encode country
categorical_df.loc[:, "country"] = label_encoder.fit_transform(categorical_df["country"])

categorical_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Label encode the 'hotel' column

# COMMAND ----------

# Label encode the 'hotel' column
categorical_df["hotel"] = label_encoder.fit_transform(categorical_df["hotel"])
categorical_df.reset_index(drop=True, inplace=True)
categorical_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Do a one-hot encode of all other categorical columns

# COMMAND ----------

# Use one-hot encoding for all other categorical columns
categorical_df = pd.get_dummies(data=categorical_df, columns=["meal", "market_segment", "distribution_channel", 
                                                              "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type", 
                                                              "reservation_status" ])
categorical_df.reset_index(drop=True, inplace=True)  
categorical_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a numerical dataframe with all the numerical variables

# COMMAND ----------

# Drop all categorical columns, which leaves us the numerical columns
numerical_df = df.drop(columns=categorical_columns, axis=1)
numerical_df.reset_index(drop=True, inplace=True)

numerical_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##Concatenate both the numerical and categorical data frames

# COMMAND ----------

# We concatenate both dataframes along the column axis
final_df = pd.concat([numerical_df, categorical_df], axis=1)
final_df.reset_index(drop=True, inplace=True)

final_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Correlation Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #####First, we will create a correlation matrix
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# Create teh correlation matrix
correlation_matrix = final_df.corr().abs()
correlation_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC #####Create the upper triangle of the correlation matrix

# COMMAND ----------

# Setting the threshold
threshold = 0.85

# COMMAND ----------

# MAGIC %md
# MAGIC This code is used to extract the upper triangle of a correlation matrix. Let's break it down step-by-step:
# MAGIC
# MAGIC - np.triu(np.ones(correlation_matrix.shape), k=1):
# MAGIC   
# MAGIC   This part creates an upper triangular matrix filled with ones. The np.triu function is a NumPy method that returns the upper triangle of a matrix.
# MAGIC
# MAGIC - The argument np.ones(correlation_matrix.shape) creates a matrix of the 
# MAGIC   same shape as the correlation_matrix but filled entirely with ones.
# MAGIC
# MAGIC - The parameter k=1 specifies that the diagonal elements and the elements 
# MAGIC   above the diagonal should be included in the upper triangle, but excludes the diagonal itself. If k=0 were used, the diagonal would also be included. As a result, this produces a matrix where the elements above the diagonal are 1 and everything else is 0.
# MAGIC
# MAGIC - .astype(bool): The .astype(bool) method converts the matrix
# MAGIC   of ones and  zeros into a Boolean matrix, where True represents the positions in the upper triangle (above the diagonal) and False represents the rest.
# MAGIC - correlation_matrix.where(...):
# MAGIC
# MAGIC   The where method of a Pandas DataFrame uses the Boolean matrix created above to filter the correlation_matrix.
# MAGIC   It retains the values of correlation_matrix in positions corresponding to True in the Boolean matrix and replaces the rest with NaN.

# COMMAND ----------

upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# COMMAND ----------

# MAGIC %md
# MAGIC #####Ensure that only one of the highly correlated features get dropped

# COMMAND ----------

# MAGIC %md
# MAGIC The code below is used to identify highly correlated features in a DataFrame and mark one of each correlated pair for removal. This is a common practice in feature selection for machine learning, where highly correlated features can introduce redundancy and multicollinearity, potentially degrading the performance of a model.
# MAGIC
# MAGIC The purpose of this code is to ensure that for any pair of highly correlated features, only one is marked for removal. This is particularly useful in scenarios where:
# MAGIC
# MAGIC You want to reduce redundancy and multicollinearity in your feature set.
# MAGIC You want to minimize overfitting in a machine learning model, as having highly correlated features can lead to a model that overly relies on certain data patterns.

# COMMAND ----------

# Ensure that only 1 of the highly correlated features get dropped
to_drop = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            column_name = correlation_matrix.columns[i]
            to_drop.append(column_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #####Show the columns that should be dropped

# COMMAND ----------

to_drop = list(set(to_drop))
to_drop

# COMMAND ----------

# MAGIC %md
# MAGIC #####Drop the columns

# COMMAND ----------

# Drop those columns
final_df.drop(columns=to_drop, inplace=True)
final_df.reset_index(drop=True, inplace=True)
final_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Create our output hotel_bookings Delta file

# COMMAND ----------

# Convert `uint8` columns to `int32`
final_df = final_df.astype({col: 'int32' for col in final_df.select_dtypes('uint8').columns})

# Remove or replace invalid characters in column names
final_df.columns = [col.replace(" ", "_").replace("/", "_") for col in final_df.columns]

# Convert Pandas DataFrame to Spark DataFrame first
spark_df = spark.createDataFrame(final_df)

# Save the Spark DataFrame as a Delta Table
delta_file = "dbfs:/FileStore/datasets/hotel_bookings.delta"
spark_df.write.format("delta").mode("overwrite").save(f"{delta_file}")


# COMMAND ----------

# MAGIC %md
# MAGIC #End of Feature Engineering
