# Databricks notebook source
import numpy as np
from databricks import feature_store as fs
from pyspark.sql.types import IntegerType, LongType, StringType, StructType, StructField

from pyspark.sql import functions as F

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# Define schema explicitly
ratings_schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("movie_id", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", LongType(), True)
    # Add more columns as needed
])

ratings_df = spark.read.option("sep", "::").option("header", "false") \
    .csv("dbfs:/FileStore/datasets/MovieLens-1M/ratings.dat", schema=ratings_schema)

# COMMAND ----------

ratings_df.display()

# COMMAND ----------

feature_store_client = fs.FeatureStoreClient()

# COMMAND ----------

#Read in the user Features
user_features_df = feature_store_client.read_table("book_ai_ml_lakehouse.movielens_features_db.user_features")
user_features_df.display()

# COMMAND ----------

# Read in the movie features
movie_features_df = feature_store_client.read_table("book_ai_ml_lakehouse.movielens_features_db.movie_features")
movie_features_df.display()

# COMMAND ----------

# Join features with ratings data
training_data = ratings_df.join(user_features_df, on='user_id').join(movie_features_df, on='movie_id')
training_data.show()

# Rename columns
training_data = (
    training_data.withColumnRenamed("scaled_avg_rating", "user_scaled_avg_rating")
      .withColumnRenamed("scaled_rating_count", "user_scaled_rating_count")
      .withColumnRenamed("scaled_avg_rating", "movie_scaled_avg_rating")
      .withColumnRenamed("scaled_rating_count", "movie_scaled_rating_count")
)

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame for training
training_data_pd = training_data.toPandas()

# Prepare the user-item matrix
user_movie_matrix = training_data_pd.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Train-test split
train_matrix, test_matrix = train_test_split(user_movie_matrix, test_size=0.2, random_state=42)

# Normalize the data (optional but can improve SVD performance)
scaler = StandardScaler()
train_matrix_scaled = scaler.fit_transform(train_matrix)
test_matrix_scaled = scaler.transform(test_matrix)

# Train a collaborative filtering model with SVD
svd = TruncatedSVD(n_components=50, random_state=42)
train_matrix_reduced = svd.fit_transform(train_matrix_scaled)

# Reconstruct the training matrix for training predictions
predicted_train_ratings = np.dot(train_matrix_reduced, svd.components_)
predicted_train_ratings = scaler.inverse_transform(predicted_train_ratings)

# Reconstruct the test matrix for predictions
test_matrix_reduced = svd.transform(test_matrix_scaled)
predicted_test_ratings = np.dot(test_matrix_reduced, svd.components_)
predicted_test_ratings = scaler.inverse_transform(predicted_test_ratings)

# Evaluation Metrics
# Flatten the matrices to calculate the error
y_true = test_matrix.values.flatten()
y_pred = predicted_test_ratings.flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# COMMAND ----------

# Classification Metrics (Convert ratings to binary for confusion matrix)
# Define a threshold to convert ratings to binary classes for evaluation
threshold = 3  # Assuming ratings >= 3 are positive (like), otherwise negative (dislike)
y_true_class = np.where(y_true >= threshold, 1, 0)
y_pred_class = np.where(y_pred >= threshold, 1, 0)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_class, y_pred_class)
print("Confusion Matrix:\n", conf_matrix)

# Visualization of the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Dislike', 'Like'], yticklabels=['Dislike', 'Like'])
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()

# Additional Metrics
print("Accuracy:", accuracy_score(y_true_class, y_pred_class))
print("Classification Report:\n", classification_report(y_true_class, y_pred_class))
