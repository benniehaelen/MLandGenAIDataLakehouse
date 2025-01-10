# Databricks notebook source
# MAGIC %run "../common/Constants"

# COMMAND ----------

BASE_HOME_CREDIT_PATH = 'dbfs:/FileStore/datasets/home_credit_default_risk'
APPLICATION_TRAIN_PATH = BASE_HOME_CREDIT_PATH + '/application_train.csv'
BUREAU_PATH = BASE_HOME_CREDIT_PATH + '/bureau.csv'
CREDIT_CARD_BALANCE_PATH = BASE_HOME_CREDIT_PATH + '/credit_card_balance.csv'
INSTALMENT_PAYMENTS_PATH = BASE_HOME_CREDIT_PATH + '/installments_payments.csv'
POS_CASH_BALANCE_PATH = BASE_HOME_CREDIT_PATH + '/POS_CASH_balance.csv'

# COMMAND ----------

# Load the main application training dataset
application_train_path = APPLICATION_TRAIN_PATH
application_train = spark.read.csv(application_train_path, header=True, inferSchema=True)

# Display the schema and a few rows
application_train.printSchema()
application_train.select("SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT").show(5)


# COMMAND ----------

from pyspark.sql.functions import col, log

# Add income-to-loan ratio
application_train = application_train.withColumn(
    "IncomeToLoanRatio", col("AMT_INCOME_TOTAL") / col("AMT_CREDIT")
)

# Log-transform income to reduce skewness
application_train = application_train.withColumn("LogIncome", log(col("AMT_INCOME_TOTAL") + 1))

# Fill missing values for family size
application_train = application_train.fillna({"CNT_FAM_MEMBERS": 0})

# Display the new features
application_train.select("SK_ID_CURR", "IncomeToLoanRatio", "LogIncome", "CNT_FAM_MEMBERS").show(5)


# COMMAND ----------

# Get a list of column names
column_names = application_train.columns

# Check for duplicate column names
duplicates = [col for col in set(column_names) if column_names.count(col) > 1]
print(f"Duplicate columns: {duplicates}")

# COMMAND ----------

# Load bureau.csv
bureau = spark.read.csv(BUREAU_PATH, header=True, inferSchema=True)

# Aggregate bureau data
from pyspark.sql.functions import count, avg, sum as _sum

# Aggregate bureau data with renamed columns
bureau_aggregated = (
    bureau.groupBy("SK_ID_CURR")
    .agg(
        count("SK_ID_BUREAU").alias("Bureau_NumberOfLoans"),
        avg("AMT_CREDIT_SUM").alias("Bureau_AvgLoanAmount"),
        _sum("AMT_CREDIT_SUM").alias("Bureau_TotalLoanAmount")
    )
)

# Join aggregated features with application_train
application_train = application_train.join(bureau_aggregated, on="SK_ID_CURR", how="left")

# Verify the schema
application_train.printSchema()


# COMMAND ----------

# Get a list of column names
column_names = application_train.columns

# Check for duplicate column names
duplicates = [col for col in set(column_names) if column_names.count(col) > 1]
print(f"Duplicate columns: {duplicates}")

# COMMAND ----------

from pyspark.sql.functions import avg, sum, col, when

# Load credit_card_balance.csv
credit_card = spark.read.csv(CREDIT_CARD_BALANCE_PATH, header=True, inferSchema=True)

# Aggregate credit card data
credit_card_aggregated = (
    credit_card.groupBy("SK_ID_CURR")
    .agg(
        avg(col("AMT_BALANCE") / col("AMT_CREDIT_LIMIT_ACTUAL")).alias("AvgCreditUtilization"),
        sum(when(col("SK_DPD") > 0, 1).otherwise(0)).alias("MissedPayments")
    )
)

# Join credit card features with application_train
application_train = application_train.join(credit_card_aggregated, on="SK_ID_CURR", how="left")


# COMMAND ----------

# Get a list of column names
column_names = application_train.columns

# Check for duplicate column names
duplicates = [col for col in set(column_names) if column_names.count(col) > 1]
print(f"Duplicate columns: {duplicates}")

# COMMAND ----------

from pyspark.sql.functions import avg, col, when

# Load installments_payments.csv
installments = spark.read.csv(INSTALMENT_PAYMENTS_PATH, header=True, inferSchema=True)

# Aggregate installments data
installments_aggregated = (
    installments.groupBy("SK_ID_CURR")
    .agg(
        avg(when(col("DAYS_INSTALMENT") >= col("DAYS_ENTRY_PAYMENT"), 1).otherwise(0)).alias("PaymentPunctuality"),
        avg(col("AMT_PAYMENT") / col("AMT_INSTALMENT")).alias("PaymentToInstallmentRatio")
    )
)

# Join installments features with application_train
application_train = application_train.join(installments_aggregated, on="SK_ID_CURR", how="left")


# COMMAND ----------

# Get a list of column names
column_names = application_train.columns

# Check for duplicate column names
duplicates = [col for col in set(column_names) if column_names.count(col) > 1]
print(f"Duplicate columns: {duplicates}")

# COMMAND ----------

from pyspark.sql.functions import sum, max, col, when

# Load POS_CASH_balance.csv
pos_cash = spark.read.csv(POS_CASH_BALANCE_PATH, header=True, inferSchema=True)

# Aggregate POS cash data
pos_cash_aggregated = (
    pos_cash.groupBy("SK_ID_CURR")
    .agg(
        sum(when(col("NAME_CONTRACT_STATUS") == "Active", 1).otherwise(0)).alias("ActiveContracts"),
        max(col("SK_DPD")).alias("MaxOverdueDays")
    )
)

# Join POS cash features with application_train
application_train = application_train.join(pos_cash_aggregated, on="SK_ID_CURR", how="left")


# COMMAND ----------

 # Get a list of column names
column_names = application_train.columns

# Check for duplicate column names
duplicates = [col for col in set(column_names) if column_names.count(col) > 1]
print(f"Duplicate columns: {duplicates}")

# COMMAND ----------

application_train.display()

# COMMAND ----------

application_train.printSchema()

# COMMAND ----------

# Get a list of column names
column_names = application_train.columns

# Check for duplicate column names
duplicates = [col for col in set(column_names) if column_names.count(col) > 1]
print(f"Duplicate columns: {duplicates}")


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS book_ai_ml_lakehouse.feature_store_db;

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

FEATURE_TABLE_NAME = "home_credit_features"
# Define the fully qualified feature table name (Unity Catalog)
feature_table_name = f"{CATALOG_NAME}.{FEATURE_STORE_DB}.{FEATURE_TABLE_NAME}"

# Create feature table and write data
fs.create_table(
    name=feature_table_name,
    primary_keys=["SK_ID_CURR"],
    schema=application_train.schema,
    description="Features derived from Home Credit Default Risk dataset"
)

fs.write_table(
    name=feature_table_name,
    df=application_train,
    mode="overwrite"
)

# COMMAND ----------

# Retrieve features from the Feature Store
training_data = fs.read_table(feature_table_name)

# Convert the Spark DataFrame to a Pandas DataFrame for Scikit-learn
training_data_pd = training_data.toPandas()


# COMMAND ----------

# Define features and target
X = training_data_pd[
    ["IncomeToLoanRatio", "LogIncome", "Bureau_NumberOfLoans", "Bureau_AvgLoanAmount", 
     "Bureau_TotalLoanAmount", "AvgCreditUtilization", "PaymentPunctuality", "ActiveContracts"]
]
y = training_data_pd["TARGET"]  # `TARGET` column indicates whether a loan default occurred


# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

from sklearn.impute import SimpleImputer

# Initialize an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy="mean")

# Apply the imputer to the feature matrix
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the RandomForestClassifier with the imputed dataset
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train_imputed, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_imputed)
y_proba = model.predict_proba(X_test_imputed)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

