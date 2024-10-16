# Databricks notebook source
# MAGIC %md
# MAGIC #Download the CIFAR10 dataset and convert to PyTorch tensor
# MAGIC This dataset can be found at: https://www.cs.toronto.edu/~kriz/cifar.html

# COMMAND ----------

import torchvision
import torchvision.transforms as transforms

# Define a series of transformations to apply to the images.
# Here, the transformation only converts images to PyTorch tensors.
# You can add other transformations such as normalization or data augmentation if needed.
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
])

# Download and load the CIFAR-10 training dataset.
# 'root' specifies where to store the downloaded dataset.
# 'train=True' indicates that we want the training set.
# 'download=True' ensures that the dataset is downloaded if not already present.
# 'transform' applies the previously defined transformation to each image in the dataset.
train_dataset = torchvision.datasets.CIFAR10(
    root='/dbfs/FileStore/datasets/CIFAR10/',  # Path to store the data in Databricks DBFS.
    train=True,                                # Indicates loading the training set.
    download=True,                             # Downloads the dataset if it doesn't exist in the path.
    transform=transform                        # Apply the transformation to convert images to tensors.
)

# Download and load the CIFAR-10 test dataset.
# Similar to the training dataset, but here 'train=False' indicates the test set.
test_dataset = torchvision.datasets.CIFAR10(
    root='/dbfs/Filestore/datasets/CIFAR10/',  # Path to store the data in Databricks DBFS.
    train=False,                               # Indicates loading the test set.
    download=True,                             # Downloads the dataset if it doesn't exist.
    transform=transform                        # Apply the transformation to convert images to tensors.
)


# COMMAND ----------

# MAGIC %md
# MAGIC #Use the resnet18 pre-trained model
# MAGIC Remove the last (classification) layer so we can use it as a feature extractor

# COMMAND ----------

import torch.nn as nn  # Import the neural network module from PyTorch
import torch  # Import PyTorch for tensor computations and model operations

# Load a pre-trained ResNet-18 model from torchvision's model zoo
# 'pretrained=True' downloads a ResNet-18 model pre-trained on the ImageNet dataset.
model = torchvision.models.resnet18(pretrained=True)

# Modify the model by removing the last fully connected classification layer.
# This is useful when you want to use the pre-trained ResNet as a feature extractor
# by removing the final layer responsible for the original ImageNet classification.
model = nn.Sequential(*list(model.children())[:-1])

# Set the model to evaluation mode.
# 'eval()' tells PyTorch to disable certain layers (like dropout and batch normalization),
# which behave differently during training and evaluation. This is necessary when using
# the model for inference or feature extraction.
model.eval()

# COMMAND ----------

# MAGIC %md
# MAGIC #Extract the image and label features

# COMMAND ----------

def extract_features(data_loader):
    # Initialize empty lists to store extracted features and labels
    features_list = []
    labels_list = []
    
    # Disable gradient calculations to improve performance during inference
    with torch.no_grad():
        # Loop over the batches of images and labels from the data_loader
        for images, labels in data_loader:
            # Pass the images through the pre-trained model to extract features
            features = model(images)
            # Flatten the feature map output to a 2D tensor where each row corresponds to an image
            features = features.view(features.size(0), -1)
            # Convert the features and labels from PyTorch tensors to NumPy arrays
            features_list.append(features.numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate the extracted features and labels from all batches into single NumPy arrays
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    # Return the feature matrix and the corresponding labels as NumPy arrays
    return features_array, labels_array



# COMMAND ----------

# MAGIC %md
# MAGIC #Setup to laod the images in batches with the pyTorch DataLoader()

# COMMAND ----------

from torch.utils.data import DataLoader  # Import the DataLoader class from PyTorch's data utility module

# Set the batch size for loading the data
batch_size = 64  # This defines the number of samples to load per batch. In this case, each batch will contain 64 images.

# Create the data loader for the training dataset
# train_loader will load the data in batches of 'batch_size' from the train_dataset.
# shuffle=False means that the data will not be shuffled before each epoch.
# If you want to randomize the order of the data during training, you would set shuffle=True.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Create the data loader for the test dataset
# test_loader works similarly to the train_loader, but it loads data from test_dataset.
# The data is not shuffled, as the test dataset should be evaluated in a fixed order.
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #Extract the training and test features using the DataLoader

# COMMAND ----------

import numpy as np  # Import NumPy for numerical operations and array manipulation

# Extract features and labels from the training set using the 'extract_features' function
# 'train_loader' is the DataLoader object created earlier, and the function will process the batches of data.
# This will return two NumPy arrays: 'train_features' containing the extracted features
# and 'train_labels' containing the corresponding labels for the training data.
train_features, train_labels = extract_features(train_loader)

# Extract features and labels from the test set using the same 'extract_features' function
# 'test_loader' is the DataLoader for the test dataset.
# This step processes the test data and returns 'test_features' (the feature matrix for the test set)
# and 'test_labels' (the labels for the test set) in NumPy array format.
test_features, test_labels = extract_features(test_loader)

# COMMAND ----------

# MAGIC %md
# MAGIC #Convert the features to a NumPy Dataframe for both train and test

# COMMAND ----------

import pandas as pd  # Import Pandas library for data manipulation and analysis

# Create a DataFrame for the training data
# 'train_features' is a NumPy array where each row contains the extracted features of a training image
# 'df_train' stores these features in a DataFrame for easier manipulation
df_train = pd.DataFrame(train_features)

# Add the 'label' column to the training DataFrame
# 'train_labels' contains the corresponding labels for the training images
# This appends the labels as a new column named 'label' in the DataFrame
df_train['label'] = train_labels

# Create a DataFrame for the test data in the same way
# 'test_features' is a NumPy array containing features extracted from the test images
df_test = pd.DataFrame(test_features)

# Add the 'label' column to the test DataFrame
# 'test_labels' contains the corresponding labels for the test images
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

