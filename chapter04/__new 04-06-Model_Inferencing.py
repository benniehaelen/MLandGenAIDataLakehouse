# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px; width:30%; height: auto;" src="https://i.imgur.com/FWzhbhX.jpeg"   />
# MAGIC
# MAGIC    Name:          chapter 04-06 Model Inferencing
# MAGIC
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-13-2024
# MAGIC
# MAGIC    Purpose:   This notebook demonstrates real-time inferencing against the deployed
# MAGIC               hotel cancellation classifier serving endpoint, and verifies that
# MAGIC               endpoint predictions are consistent with the registered model.
# MAGIC
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Load the feature table and prepare test data
# MAGIC         2 - Retrieve workspace credentials
# MAGIC         3 - Define the endpoint scoring function
# MAGIC         4 - Call the endpoint and compare against the registered model
# MAGIC         5 - Evaluate accuracy and prediction distribution

# COMMAND ----------

import os
import json
import time
import mlflow
import mlflow.sklearn
import requests
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog coordinates
# MAGIC
# MAGIC All sections of this notebook reference the same catalog, schema, model name,
# MAGIC and endpoint name used in the model building and deployment notebooks.
# MAGIC Update these values if your setup uses different names.

# COMMAND ----------

catalog       = "book_ai_ml_lakehouse"
schema        = "default"
model_name    = "hotel_cancellation_classifier"
model_uc_path = f"{catalog}.{schema}.{model_name}"
endpoint_name = "hotel-cancellation-classifier"

print(f"Model:    {model_uc_path}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the feature table and prepare test data
# MAGIC
# MAGIC We read the Unity Catalog feature table and apply the same 80/20 split and
# MAGIC scaling used in the model building notebook. Using identical parameters
# MAGIC guarantees that the test rows sent to the endpoint are the same held-out
# MAGIC records the model has never seen during training.

# COMMAND ----------

spark_df    = spark.read.table(f"{catalog}.{schema}.hotel_bookings_features")
bookings_df = spark_df.toPandas()

X = bookings_df.drop(columns=["is_canceled"])
y = bookings_df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler        = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert to a named DataFrame so column names are preserved
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

print(f"Test set ready: {X_test_scaled_df.shape[0]:,} rows, {X_test_scaled_df.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve workspace credentials
# MAGIC
# MAGIC We retrieve the workspace URL and authentication token from the Databricks
# MAGIC notebook context. This avoids hardcoding credentials and keeps the notebook
# MAGIC portable across workspaces and team members.
# MAGIC
# MAGIC If your organisation requires using a secret scope instead, replace the token
# MAGIC line with:
# MAGIC `token = dbutils.secrets.get(scope="<scope>", key="<key>")`

# COMMAND ----------

workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
token         = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type":  "application/json"
}

score_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
print(f"Endpoint URL: {score_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the endpoint scoring function
# MAGIC
# MAGIC The `score_endpoint()` function sends a batch of rows to the serving endpoint
# MAGIC and returns the predictions. It uses the `dataframe_records` payload format,
# MAGIC consistent with the deployment notebook. It also measures and returns the
# MAGIC round-trip latency so we can validate that the endpoint meets response time
# MAGIC expectations.

# COMMAND ----------

def score_endpoint(df_sample, headers, score_url):
    """
    Send a pandas DataFrame to the serving endpoint and return predictions.

    Parameters
    ----------
    df_sample  : pd.DataFrame  Rows to score (already scaled, named columns)
    headers    : dict          Authorization and content-type headers
    score_url  : str           Full invocations URL for the endpoint

    Returns
    -------
    predictions : list    Predicted class labels (0 or 1)
    latency_ms  : float   Round-trip time in milliseconds
    """
    payload   = {"dataframe_records": df_sample.to_dict(orient="records")}
    data_json = json.dumps(payload, allow_nan=True)

    start    = time.time()
    response = requests.post(url=score_url, headers=headers, data=data_json)
    latency  = (time.time() - start) * 1000

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}: {response.text}"
        )

    return response.json().get("predictions", []), round(latency, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the registered Champion model for local comparison
# MAGIC
# MAGIC Rather than training a new local model from scratch, we load the same
# MAGIC registered model that the endpoint is serving, using the `@Champion` alias.
# MAGIC This guarantees that the local and endpoint predictions should be identical,
# MAGIC making any discrepancy a meaningful signal rather than an expected difference
# MAGIC between two different model types.

# COMMAND ----------

client   = MlflowClient()
versions = client.search_model_versions(f"name='{model_uc_path}'")
latest_version = max(versions, key=lambda v: int(v.version)).version

alias_uri     = f"models:/{model_uc_path}@Champion"
champion_model = mlflow.sklearn.load_model(alias_uri)

print(f"Loaded {type(champion_model).__name__} from {alias_uri}")
print(f"(Version {latest_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the endpoint and compare against the registered model
# MAGIC
# MAGIC We send 25 rows from the test set to the live endpoint, then run the same
# MAGIC rows through the locally loaded Champion model. The two prediction columns
# MAGIC should be identical. Any mismatch would indicate a serialization or
# MAGIC environment discrepancy in the serving layer worth investigating.

# COMMAND ----------

num_predictions = 25
sample          = X_test_scaled_df.head(num_predictions)
y_test_sample   = y_test.iloc[:num_predictions].reset_index(drop=True)

# Call the endpoint
endpoint_preds, latency_ms = score_endpoint(sample, headers, score_url)
print(f"Endpoint response time: {latency_ms} ms")

# Local Champion model predictions on the same rows
local_preds = champion_model.predict(sample)

# Build comparison DataFrame including ground truth
pred_df = pd.DataFrame({
    "Ground truth":              y_test_sample,
    "Local model prediction":    local_preds,
    "Endpoint prediction":       endpoint_preds
})

print(f"\nPrediction comparison (first {num_predictions} test rows):")
print(pred_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify prediction consistency

# COMMAND ----------

match = (pred_df["Local model prediction"] == pred_df["Endpoint prediction"]).all()

if match:
    print("All endpoint predictions match the local Champion model.")
else:
    n_diff = (pred_df["Local model prediction"] != pred_df["Endpoint prediction"]).sum()
    print(f"WARNING: {n_diff} of {num_predictions} predictions differ between the endpoint and local model.")
    print("Check the serving environment version and model artifact for discrepancies.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate accuracy and prediction distribution
# MAGIC
# MAGIC We compare both the local model and the endpoint against the ground truth
# MAGIC labels to confirm consistent accuracy, and inspect the prediction distribution
# MAGIC to verify that cancellation and non-cancellation rates are in line with the
# MAGIC training data baseline.

# COMMAND ----------

local_acc    = accuracy_score(y_test_sample, pred_df["Local model prediction"])
endpoint_acc = accuracy_score(y_test_sample, pred_df["Endpoint prediction"])

print(f"Local model accuracy   (n={num_predictions}): {local_acc:.4f}")
print(f"Endpoint accuracy      (n={num_predictions}): {endpoint_acc:.4f}")

# Prediction distribution
print(f"\nEndpoint prediction distribution:")
dist = pred_df["Endpoint prediction"].value_counts().sort_index()
for label, count in dist.items():
    meaning = "not canceled" if label == 0 else "canceled"
    print(f"  {label} ({meaning}): {count} ({count/num_predictions*100:.0f}%)")

print(f"\nClassification report (endpoint predictions vs ground truth):")
print(classification_report(y_test_sample, pred_df["Endpoint prediction"],
                             target_names=["not canceled", "canceled"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## End of notebook
