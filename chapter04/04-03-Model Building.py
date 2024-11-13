# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px;" src="https://learning.oreilly.com/covers/urn:orm:book:9781098139711/400w/" />  
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 03-03-Model Buidling
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-09-2024
# MAGIC
# MAGIC    Purpose:   This notebook performs the model creation, training and evaluation for the Hotel Bookings use case
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Read the hotel-booking delta file as a Pandas dataframe
# MAGIC         2 - Prepare our test and train data
# MAGIC               2-1 Define the X and y variables
# MAGIC               2-2 Split the data into training and test data
# MAGIC               2-3 Apply standard scaling to our X variables
# MAGIC               2-4 Visualize the standard deviation and mean
# MAGIC               2-5 Fill in default value for the 'agent' column
# MAGIC         3 - Model Building
# MAGIC               3-1 Set the MLFlow experiment
# MAGIC               3-1 Logistic Regression
# MAGIC               3-2 K-Neighboards Classifier
# MAGIC               3-3 Decision Tree Classifier
# MAGIC               3-4 Random Forest Classifier
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade threadpoolctl==3.5.0

# COMMAND ----------

# MAGIC %restart_python

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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# COMMAND ----------

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
# MAGIC

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

# MAGIC %md
# MAGIC ##Apply standard scaling to our X variables
# MAGIC This will get the data to a mean of zero and a standard deviation of one

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

X_test_scaled

# COMMAND ----------

# MAGIC %md
# MAGIC ##Visualize the Standard Deviation and Mean

# COMMAND ----------

# Convert X_train_scaled back to a DataFrame with the original feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Step 1: Calculate variance and correlation of features
# Calculate variance for each feature in the scaled training dataset
# Variance measures the dispersion of data points. High variance indicates
# that a feature might have a stronger impact on the model.
feature_variance = X_train_scaled_df.var()

# Calculate the correlation of each feature with the target variable
# Correlation measures the linear relationship between a feature and the target.
# Features with high absolute correlation values are more predictive.
feature_correlation = X_train_scaled_df.corrwith(y_train)

# Step 2: Select top 5 features based on variance and correlation
# Select the top 5 features with the highest variance
top_variance_features = feature_variance.nlargest(5).index

# Select the top 5 features with the highest absolute correlation with the target
top_correlation_features = feature_correlation.abs().nlargest(5).index

# Step 3: Combine the selected features into a single list
# Use a set to avoid duplicates, then convert back to a list.
# This ensures that if a feature is selected based on both variance and correlation, it is only included once.
selected_features = list(set(top_variance_features).union(set(top_correlation_features)))

# Step 4: Visualize the selected features using density plots
# Determine the number of features to visualize
num_features = len(selected_features)

# Calculate the number of rows and columns for subplots
# Use 2 columns for visualization, and calculate the number of rows dynamically
rows = (num_features // 2) + 1  # If num_features is odd, add one more row
cols = 2                        # Fixed number of columns

# Set the overall figure size, with height adjusted based on the number of rows
plt.figure(figsize=(16, 4 * rows))

# Loop through each selected feature and create a density plot
for i, feature in enumerate(selected_features):
    # Create a subplot for each feature
    plt.subplot(rows, cols, i + 1)  # Position subplots in a grid layout
    sns.kdeplot(X_train_scaled_df[feature], label='Scaled', color='orange', fill=True)
    
    # Set the title of each subplot, with adjusted font size and padding to prevent overlap
    plt.title(f'Density Plot of {feature} (Scaled)', fontsize=14, pad=20)
    
    # Plot a vertical line indicating the mean value of the feature
    plt.axvline(X_train_scaled_df[feature].mean(), color='red', linestyle='--', label='Mean')
    
    # Set x-axis label for each feature plot
    plt.xlabel(feature, fontsize=12)
    
    # Display legend on each subplot
    plt.legend()

# Adjust the spacing between subplots to ensure clarity
plt.tight_layout(pad=3.0)  # Add padding to prevent titles and plots from overlapping
plt.show()  # Display the combined plot




# COMMAND ----------

# MAGIC %md
# MAGIC #Model Building

# COMMAND ----------

# MAGIC %md
# MAGIC ##Set the MLflow Experiment

# COMMAND ----------


# Set the MLflow experiment
# This command sets the current experiment where all the MLflow runs will be logged.

# The 'experiment_name' argument specifies the location of the experiment within the Databricks workspace.
# In this case, the experiment is named 'hotel_bookings_cancellations_prediction' and is stored
# under the user's directory '/Users/bhaelen2018@outlook.com/' in the Databricks workspace.

# MLflow will use this experiment to organize and track different model runs, metrics, parameters,
# and artifacts (such as models) that you log during the machine learning workflow.
mlflow.set_experiment(experiment_name = '/Users/bhaelen2018@outlook.com/hotel_bookings_cancellations_prediction')


# COMMAND ----------

# MAGIC %md
# MAGIC ##Logistic Regression

# COMMAND ----------

from mlflow.models import infer_signature

# Convert y_train and y_test to NumPy arrays before passing them to the model
y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()

# Enable automatic logging of parameters, metrics, and model for scikit-learn models
mlflow.sklearn.autolog()

# Start a new MLflow run and log the model
with mlflow.start_run(run_name='logistic_regression_model') as run1:

    # Step 1: Train your model 
    model = LogisticRegression(max_iter=10000, class_weight='balanced')
    model.fit(X_train_scaled, y_train_array)

    # Step 2: Make predictions on the test set
    y_pred_lr = model.predict(X_test_scaled)

    # Step 3: Calculate performance metrics
    acc_lr = accuracy_score(y_test_array, y_pred_lr)
    conf_matrix = confusion_matrix(y_test_array, y_pred_lr)
    clf_report = classification_report(y_test_array, y_pred_lr)

    # Step 4: Infer the model signature from the input and output data
    signature = infer_signature(X_train_scaled, y_pred_lr)

    # Log the model with the inferred signature
    model_name = "Logistic_Regression_Model"
    mlflow.sklearn.log_model(model, model_name, signature=signature)

    # Step 5: Print out the model's performance
    print(f"Accuracy Score of Logistic Regression: {acc_lr:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{clf_report}")

    # Access and print the ID of the active MLflow run
    run1 = mlflow.active_run()
    print("Active run ID:", run1.info.run_id)



# COMMAND ----------

# MAGIC %md
# MAGIC ##K-Neighbors Classifier

# COMMAND ----------



# Start a new MLflow run and give it a descriptive name ('k_neighbors_classifier') for easy identification
with mlflow.start_run(run_name='k_neighbors_classifier') as run2:

    # Initialize the K-Nearest Neighbors classifier with explicit hyperparameters
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')  # Feel free to adjust parameters

    # Train the KNN model on the scaled training data
    knn.fit(X_train_scaled, y_train_array)

    # Use the trained model to make predictions on the scaled test data
    y_pred_knn = knn.predict(X_test_scaled)

    # Infer the signature from the input and the model's predictions
    signature = infer_signature(X_train_scaled, y_pred_knn)

    # Provide an input example for MLflow to infer the model signature (use a small subset of the training data)
    input_example = X_train_scaled[:5]  # Using the first 5 rows of the training data as an example

    # Log the KNN model to MLflow with the inferred signature and input example
    mlflow.sklearn.log_model(knn, "knn_model", signature=signature, input_example=input_example)

    # Calculate performance metrics
    acc_knn = accuracy_score(y_test_array, y_pred_knn)
    conf = confusion_matrix(y_test_array, y_pred_knn)
    clf_report = classification_report(y_test_array, y_pred_knn)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", acc_knn)
    mlflow.log_metrics({
        "precision": precision_score(y_test_array, y_pred_knn, average='weighted'),
        "recall": recall_score(y_test_array, y_pred_knn, average='weighted'),
        "f1_score": f1_score(y_test_array, y_pred_knn, average='weighted')
    })

    # Log the confusion matrix as an artifact (optional but useful)
    mlflow.log_text(str(conf), "confusion_matrix.txt")

    # Log the classification report
    mlflow.log_text(clf_report, "classification_report.txt")

    # Print out the performance metrics
    print(f"Accuracy Score of KNN: {acc_knn:.4f}")
    print(f"Confusion Matrix:\n{conf}")
    print(f"Classification Report:\n{clf_report}")

    # Access and print the ID of the active MLflow run
    run2 = mlflow.active_run()
    print("Active run ID:", run2.info.run_id)



# COMMAND ----------

# MAGIC %md
# MAGIC ##Decision Tree Classifier

# COMMAND ----------

# Start a new MLflow run with a descriptive name
with mlflow.start_run(run_name='decision_tree_classifier') as run3:

    # Initialize the Decision Tree classifier with some hyperparameters
    dtc = DecisionTreeClassifier(max_depth=5, random_state=42)  # Adjust hyperparameters as needed
    
    # Train the Decision Tree model on the scaled training data
    dtc.fit(X_train_scaled, y_train_array)

    # Make predictions on the scaled test data
    y_pred_dtc = dtc.predict(X_test_scaled)

    # Calculate performance metrics
    acc_dtc = accuracy_score(y_test_array, y_pred_dtc)
    conf = confusion_matrix(y_test_array, y_pred_dtc)
    clf_report = classification_report(y_test_array, y_pred_dtc)

    # Infer the model signature based on input and output
    signature = infer_signature(X_train_scaled, y_pred_dtc)

    # Log performance metrics to MLflow
    mlflow.log_metric("accuracy", acc_dtc)
    mlflow.log_metrics({
        "precision": precision_score(y_test_array, y_pred_dtc, average='weighted'),
        "recall": recall_score(y_test_array, y_pred_dtc, average='weighted'),
        "f1_score": f1_score(y_test_array, y_pred_dtc, average='weighted')
    })

    # Log the confusion matrix and classification report as text artifacts
    mlflow.log_text(str(conf), "confusion_matrix.txt")
    mlflow.log_text(clf_report, "classification_report.txt")

    # Log the Decision Tree model to MLflow with the inferred signature
    mlflow.sklearn.log_model(dtc, "decision_tree_model", signature=signature)

    # Print out the performance metrics
    print(f"Accuracy Score of Decision Tree: {acc_dtc:.4f}")
    print(f"Confusion Matrix:\n{conf}")
    print(f"Classification Report:\n{clf_report}")

    # Access and print the ID of the active MLflow run
    run3 = mlflow.active_run()
    print("Active run ID:", run3.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Random Forest Classifier

# COMMAND ----------

# Start a new MLflow run with a descriptive name
with mlflow.start_run(run_name='random_forest_classifier') as run4:
    
    # Initialize the RandomForestClassifier with explicit hyperparameters
    rd_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Adjust hyperparameters as needed
    
    # Train the Random Forest classifier on the scaled training data
    rd_clf.fit(X_train_scaled, y_train_array)

    # Make predictions on the scaled test data
    y_pred_rd_clf = rd_clf.predict(X_test_scaled)

    # Infer the signature based on the input features and model predictions
    signature = infer_signature(X_train_scaled, y_pred_rd_clf)

    # Calculate performance metrics
    acc_rd_clf = accuracy_score(y_test_array, y_pred_rd_clf)
    conf = confusion_matrix(y_test_array, y_pred_rd_clf)
    clf_report = classification_report(y_test_array, y_pred_rd_clf)

    # Log the Random Forest model to MLflow with the inferred signature
    mlflow.sklearn.log_model(rd_clf, "random_forest_model", signature=signature)

    # Log performance metrics to MLflow
    mlflow.log_metric("accuracy", acc_rd_clf)
    mlflow.log_metrics({
        "precision": precision_score(y_test_array, y_pred_rd_clf, average='weighted'),
        "recall": recall_score(y_test_array, y_pred_rd_clf, average='weighted'),
        "f1_score": f1_score(y_test_array, y_pred_rd_clf, average='weighted')
    })

    # Log the confusion matrix and classification report as text artifacts in MLflow
    mlflow.log_text(str(conf), "confusion_matrix.txt")
    mlflow.log_text(clf_report, "classification_report.txt")

    # Print out the performance metrics for easy access in the notebook
    print(f"Accuracy Score of Random Forest: {acc_rd_clf:.4f}")
    print(f"Confusion Matrix:\n{conf}")
    print(f"Classification Report:\n{clf_report}")

    # Access and print the ID of the active MLflow run
    run4 = mlflow.active_run()
    print("Active run ID:", run3.info.run_id)

# COMMAND ----------

from sklearn.decomposition import PCA

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_train_scaled)

# Create a DataFrame with the PCA components
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])

# Plot PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.5)
plt.title('PCA Plot of Scaled Features', fontsize=16)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# COMMAND ----------

# Extract principal components (eigenvectors)
principal_components = pca.components_

# Create a DataFrame to show the principal components
components_df = pd.DataFrame(principal_components, columns=X_train.columns, index=[f'PC{i+1}' for i in range(principal_components.shape[0])])

# Print principal components
print("Principal Component Loadings (Eigenvectors):")
print(components_df)

# COMMAND ----------

# Create a Linear Regression instance
lr = LinearRegression()



# Fit the model
lr.fit(X_train_scaled, y_train)

# Predict using the test set
y_pred =  lr.predict(X_test_scaled)

# Calculate and display metrics
testing_score = r2_score(y_test, y_pred)
mean_absolute_score = mean_absolute_error(y_test, y_pred)
mean_sq_error = mean_squared_error(y_test, y_pred)

# Display results
print(f"R² Score: {testing_score:.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_score:.4f}")
print(f"Mean Squared Error (MSE): {mean_sq_error:.4f}")

# COMMAND ----------

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

y_pred =  knn.predict(X_test)

testing_score = r2_score(y_test, y_pred)
mean_absolute_score = mean_absolute_error(y_test, y_pred)
mean_sq_error = mean_squared_error(y_test, y_pred)

# Display results
print(f"R² Score: {testing_score:.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_score:.4f}")
print(f"Mean Squared Error (MSE): {mean_sq_error:.4f}")



# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)

acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)
conf = confusion_matrix(y_test, y_pred_rd_clf)
clf_report = classification_report(y_test, y_pred_rd_clf)

print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

acc_dtc = accuracy_score(y_test, y_pred_dtc)
conf = confusion_matrix(y_test, y_pred_dtc)
clf_report = classification_report(y_test, y_pred_dtc)

print(f"Accuracy Score of Decision Tree is : {acc_dtc}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# COMMAND ----------

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred =  lr.predict(X_test)

testing_score = r2_score(y_test, y_pred)
mean_absolute_score = mean_absolute_error(y_test, y_pred)
mean_sq_error = mean_squared_error(y_test, y_pred)

# COMMAND ----------

mean_absolute_score

# COMMAND ----------

mean_sq_error

# COMMAND ----------

testing_score


# COMMAND ----------

plt.figure(figsize = (24, 12))

corr = df.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()

# COMMAND ----------

import plotly.express as px

# Create interactive heatmap using plotly
fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
fig.update_layout(title='Interactive Correlation Heatmap', width=800, height=800)
fig.show()

# COMMAND ----------

# Filter for strong correlations only (absolute value > 0.7)
strong_corr = correlation_matrix[(correlation_matrix.abs() > 0.7) & (correlation_matrix != 1.0)].dropna(how='all', axis=0).dropna(how='all', axis=1)

# Plot heatmap of strong correlations
plt.figure(figsize=(12, 10))
sns.heatmap(strong_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap of Strong Correlations (|corr| > 0.7)")
plt.show()

# COMMAND ----------

# Create a clustered heatmap
plt.figure(figsize=(12, 10))
sns.clustermap(correlation_matrix, cmap='coolwarm', figsize=(12, 12), annot=False)
plt.title("Clustered Heatmap of Correlations")
plt.show()

# COMMAND ----------

# Unstack and sort the correlation pairs
sorted_corr_pairs = correlation_matrix.unstack().sort_values(kind="quicksort", ascending=False)

# Select the top N correlations (excluding 1.0 correlations)
top_n_corr = sorted_corr_pairs[sorted_corr_pairs < 1].nlargest(20)

# Plot top N correlations
plt.figure(figsize=(10, 6))
top_n_corr.plot(kind='bar', color='skyblue')
plt.title("Top 20 Most Correlated Feature Pairs")
plt.ylabel("Correlation Coefficient")
plt.show()

# COMMAND ----------

# Visualize missing data using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(final)df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# COMMAND ----------

# Handle missing values
# For 'company' and 'agent', fill missing values with 0 (assuming missing means no company/agent)
df['company'].fillna(0, inplace=True)
df['agent'].fillna(0, inplace=True)

# For 'children', fill missing values with 0 (assuming missing means no children)
df['children'].fillna(0, inplace=True)


# COMMAND ----------

# Verify that missing values have been handled
print("Total missing values after imputation:", df.isnull().sum().sum())

# COMMAND ----------

# Check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())


# COMMAND ----------

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

# COMMAND ----------

# Drop irrelevant columns
# Assuming 'company' and 'agent' are not useful for this analysis
df.drop(['company', 'agent', 'reservation_status_date'], axis=1, inplace=True)

# COMMAND ----------

# Convert 'reservation_status' to datetime if needed (not required here)
# Convert 'arrival_date_year' to string if treating it as a categorical variable
df['arrival_date_year'] = df['arrival_date_year'].astype(str)


# COMMAND ----------

# Convert 'children', 'babies', 'adults' to integer types
df['children'] = df['children'].astype(int)
df['babies'] = df['babies'].astype(int)
df['adults'] = df['adults'].astype(int)


# COMMAND ----------

# Identify categorical variables
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features)

# COMMAND ----------

# One-Hot Encoding for nominal categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# COMMAND ----------

# Create new feature 'total_guests' as the sum of adults, children, and babies
df_encoded['total_guests'] = df_encoded['adults'] + df_encoded['children'] + df_encoded['babies']

# COMMAND ----------

# Create new feature 'total_nights' as the sum of stays_in_weekend_nights and stays_in_week_nights
df_encoded['total_nights'] = df_encoded['stays_in_weekend_nights'] + df_encoded['stays_in_week_nights']

# COMMAND ----------

# Plot 'adr' to identify outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x=df_encoded['adr'])
plt.title('Boxplot of ADR')
plt.show()

# COMMAND ----------

# Remove outliers in 'adr' where values are excessively high
# Define an upper limit (e.g., 99th percentile)
upper_limit = df_encoded['adr'].quantile(0.99)
df_encoded = df_encoded[df_encoded['adr'] <= upper_limit]

# COMMAND ----------

# Separate features and target variable
X = df_encoded.drop('adr', axis=1)
y = df_encoded['adr']

# COMMAND ----------

# Identify numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical features
X[numerical_features] = scaler.fit_transform(X[numerical_features])


# COMMAND ----------

fig1, ax = plt.subplots(figsize = (12, 10))

sns.heatmap(df_encoded.corr(), annot = True)

plt.show()

# COMMAND ----------


