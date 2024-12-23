# Databricks notebook source
spark._jvm.java.lang.System.getProperty("java.version")


# COMMAND ----------

# Get the list of JARs as a string
jars = spark._jsc.sc().listJars().toString()

# Filter JARs that contain 'jdbc' in their names
jdbc_drivers = [jar.strip() for jar in jars.split(",") if 'jdbc' in jar.lower()]

print("JDBC Drivers in the Runtime:")
for driver in jdbc_drivers:
    print(driver)



# COMMAND ----------

# MAGIC %sh
# MAGIC find /databricks -name "*.jar" | grep -i "jdbc"
# MAGIC

# COMMAND ----------


