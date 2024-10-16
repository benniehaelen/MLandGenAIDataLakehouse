# Databricks notebook source
import torchvision
import torchvision.transforms as transforms

# Define a transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load the training data
train_dataset = torchvision.datasets.CIFAR10(
    root='/dbfs/FileStore/datasets/CIFAR10/',
    train=True,
    download=True,
    transform=transform
)

# Download and load the test data
test_dataset = torchvision.datasets.CIFAR10(
    root='/dbfs:/Filestore/datasets/CIFAR10/',
    train=False,
    download=True,
    transform=transform
)


# COMMAND ----------

import torch.nn as nn
import torch

# Load pre-trained ResNet18 model
model = torchvision.models.resnet18(pretrained=True)

# Remove the last classification layer
model = nn.Sequential(*list(model.children())[:-1])

# Set the model to evaluation mode
model.eval()



# COMMAND ----------

def extract_features(data_loader):
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            # Extract features
            features = model(images)
            # Flatten the features
            features = features.view(features.size(0), -1)
            # Convert to NumPy arrays
            features_list.append(features.numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate all batches
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    return features_array, labels_array


# COMMAND ----------

from torch.utils.data import DataLoader

# Create data loaders
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# COMMAND ----------

import numpy as np

# Extract features and labels from the training set
train_features, train_labels = extract_features(train_loader)

# Extract features and labels from the test set
test_features, test_labels = extract_features(test_loader)


# COMMAND ----------

import pandas as pd

# Create DataFrames for training data
df_train = pd.DataFrame(train_features)
df_train['label'] = train_labels

# Create DataFrames for test data
df_test = pd.DataFrame(test_features)
df_test['label'] = test_labels


# COMMAND ----------

# Combine training and test data
df_full = pd.concat([df_train, df_test], ignore_index=True)


# COMMAND ----------

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df_full)


# COMMAND ----------

from databricks import automl


# COMMAND ----------

# Specify the target column
target_col = 'label'

# Start AutoML classification
summary = automl.classify(dataset=spark_df, target_col=target_col, timeout_minutes=60)

