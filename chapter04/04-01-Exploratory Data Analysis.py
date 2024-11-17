# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src= "https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"/>&nbsp;&nbsp;<font size="16"><b>AI, ML and GenAI in the Lakehouse<b></font></span>
# MAGIC <img style="float: left; margin: 0px 15px 15px 0px;" src="https://learning.oreilly.com/covers/urn:orm:book:9781098139711/400w/" />  
# MAGIC
# MAGIC
# MAGIC  
# MAGIC   
# MAGIC    Name:          chapter 03-01-Exploratory Data Analysis
# MAGIC  
# MAGIC    Author:    Bennie Haelen
# MAGIC    Date:      10-11-2024
# MAGIC
# MAGIC    Purpose:   This notebook performs the exploratory data analysis of the hotel booking dataset for chapter 3 of the book: Intro to ML on Databricks
# MAGIC                  
# MAGIC       An outline of the different sections in this notebook:
# MAGIC         1 - Read the hotel-booking.csv notebook and display key statistics
# MAGIC         2 - Exploratory Data Analysis
# MAGIC               2-1 Question: From what country originate mosts of the guests?
# MAGIC               2-2 Question: How much do guests pay per room per night?
# MAGIC               2-3 Question: How does the price vary per night over the year?
# MAGIC               2-3 Question: Which are the busy months?
# MAGIC               2-4 Question: How long do people stay at the hotels?
# MAGIC          3 - Assumptions
# MAGIC               3-1 The cancelation rate of city hotels is higher than resort hotels.
# MAGIC               3-2 The earlier the booking made, higher the chances of cancellation.
# MAGIC               3-3 Bookings for longer durations have lower cancellations
# MAGIC               3-4 A repeated guest is less likely to cancel current booking.
# MAGIC               3-5 Higher previous cancellations lead to cancellation of current bookings.
# MAGIC               3-6 If room assigned is not the reserved room type, customer might cancel.
# MAGIC               3-7 If # of booking changes made is high, chance of cancellation is low.
# MAGIC               3-8 Refundable bookings or those without deposit have higher cancellations.
# MAGIC               3-9 If the # of days in waiting list is high, cancelations are higher
# MAGIC    
# MAGIC

# COMMAND ----------

# MAGIC %pip install folium==0.17.0
# MAGIC %pip install sort-dataframeby-monthorweek==0.4
# MAGIC %pip install sorted-months-weekdays==0.2

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC #####Perform the required imports
# MAGIC The primary libraries that we use are:
# MAGIC - NumPy for analytic arrays
# MAGIC - Pandas for DataFrames
# MAGIC - MatplotLib for generating plots
# MAGIC - Folium for Earth maps
# MAGIC - Plotly for advanced plots
# MAGIC - Scikit-learn for all machine learning tasks

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import folium
from folium.plugins import HeatMap
import plotly.express as px

# COMMAND ----------

# MAGIC %md
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

# The df.info() function provides a concise summary of the DataFrame.
# It is particularly useful for a quick inspection of the data, its structure, and the memory usage.
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #Start of Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Question: From what country originate mosts of the guests?

# COMMAND ----------

# Filter the DataFrame to only include rows where bookings were not canceled (is_canceled == 0),
# and then count the occurrences of each country in the 'country' column.
# 'value_counts()' counts the frequency of each country in the filtered dataset.
guests_by_country = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()

# Rename the columns in the resulting DataFrame:
# 'index' becomes 'country' and the count values become 'No of guests'.
guests_by_country.columns = ['country', 'No of guests']

# Display the final DataFrame containing countries and the number of guests from each country
guests_by_country


# COMMAND ----------

# MAGIC %md
# MAGIC Create a plot

# COMMAND ----------

# Initialize a base map using Folium. 
# This creates an empty map object that can be used to add various map elements (e.g., markers, layers, etc.).
# However, in this specific case, the folium map is not directly used in conjunction with the Plotly map.
basemap = folium.Map()

# Create a choropleth map using Plotly Express (px) to visualize the distribution of guests by country.
# This map will color each country based on the number of guests.
# - 'guests_by_country' is the DataFrame that contains the country names and the number of guests per country.
# - 'locations' takes the country codes or names from the 'country' column and uses them to map each country on the world map.
# - 'color' controls the intensity of the color based on the values in the 'No of guests' column, where higher values will have darker colors.
# - 'hover_name' allows the country name to be shown when the mouse hovers over the country, improving interactivity.
# - 'color_continuous_scale' sets the color scheme for the map, in this case, using the 'Viridis' scale (a gradient from blue to yellow to green).
guests_map = px.choropleth(guests_by_country, 
                           locations=guests_by_country['country'], 
                           color=guests_by_country['No of guests'], 
                           hover_name=guests_by_country['country'],
                           color_continuous_scale='Viridis')  # Viridis is a perceptually uniform colormap

# Customize the appearance of the map to enhance visual clarity.
# - 'showcoastlines' will display the coastlines of the continents in the map.
# - 'coastlinecolor' sets the color of the coastlines to black for better contrast.
# - 'showland' and 'landcolor' will display the land areas in a light gray color, making the countries stand out more.
# - 'showocean' and 'oceancolor' will display the ocean in a light blue color to improve the overall map's aesthetics.
# - 'showlakes' and 'lakecolor' will add lakes to the map and color them blue to differentiate them from land areas.
guests_map.update_geos(showcoastlines=True, coastlinecolor="Black",
                       showland=True, landcolor="lightgray",
                       showocean=True, oceancolor="lightblue",
                       showlakes=True, lakecolor="blue")

# Render and display the choropleth map within the notebook or output.
# This method outputs the map for visualization in an interactive format, allowing for zooming, panning, and hover effects.
guests_map.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ##Question: How much do guests pay per room per night?

# COMMAND ----------

# Filter the DataFrame to include only rows where bookings were not canceled (is_canceled == 0).
# This results in a subset of the data for guests who actually stayed in the hotel.
data = df[df['is_canceled'] == 0]

# Create a box plot using Plotly Express (px) to visualize the distribution of 'adr' (average daily rate) 
# for different room types, while coloring the plot by hotel type ('hotel' column).
# - 'data_frame' is the DataFrame passed to the plot, which is the non-canceled bookings (data).
# - 'x' represents the categorical variable 'reserved_room_type' (the type of room reserved by the guest).
# - 'y' is the numerical variable 'adr' (average daily rate), which shows the price per night.
# - 'color' divides the plot based on the 'hotel' type (categorizing by hotel types, likely "city" or "resort").
# - 'template' applies a predefined 'seaborn' theme for a light grey background and 
#    muted colors for a softer visual look.
px.box(data_frame=data, x='reserved_room_type', y='adr', color='hotel', template='seaborn')


# COMMAND ----------

# MAGIC %md
# MAGIC ##Question: How does the price vary per night over the year?

# COMMAND ----------

# MAGIC %md
# MAGIC ###Get the City and Resort hotels where bookings were not canceled

# COMMAND ----------

# Filter the DataFrame to get only the data for the "Resort Hotel" where bookings were not canceled.
# - 'df['hotel'] == 'Resort Hotel'' ensures that only rows related to the Resort Hotel are selected.
# - 'df['is_canceled'] == 0' ensures that only bookings that were not canceled (is_canceled == 0) are included.
# The resulting 'data_resort' DataFrame contains information about non-canceled Resort Hotel bookings.
data_resort = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]

# Filter the DataFrame to get only the data for the "City Hotel" where bookings were not canceled.
# - 'df['hotel'] == 'City Hotel'' ensures that only rows related to the City Hotel are selected.
# - 'df['is_canceled'] == 0' ensures that only bookings that were not canceled (is_canceled == 0) are included.
# The resulting 'data_city' DataFrame contains information about non-canceled City Hotel bookings.
data_city = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 0)]

# COMMAND ----------

# MAGIC %md
# MAGIC ###Get the average daily rate for each month for the Resort Hotel

# COMMAND ----------

# Group the 'data_resort' DataFrame by the 'arrival_date_month' column to calculate the mean ADR (Average Daily Rate) for each month.
# - 'groupby(['arrival_date_month'])' groups the data by month, treating each unique value in 'arrival_date_month' as a separate group.
# - 'adr' is the column representing the average daily rate, and 'mean()' calculates the mean ADR for each month.
# - 'reset_index()' converts the result of the groupby operation back into a DataFrame, so it can be displayed and manipulated more easily.
resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()

# Display the result, which is a DataFrame with each month and the corresponding mean ADR for Resort Hotels.
resort_hotel

# COMMAND ----------

# MAGIC %md
# MAGIC ###Get the average daily rate for each month for the City Hotel

# COMMAND ----------

# Group the 'data_city' DataFrame by the 'arrival_date_month' column to calculate the mean ADR (Average Daily Rate) for each month.
# - 'groupby(['arrival_date_month'])' groups the data by the month of arrival, so each unique month is treated as a group.
# - 'adr' refers to the average daily rate, and 'mean()' calculates the mean ADR for each month.
# - 'reset_index()' converts the result of the groupby operation back into a regular DataFrame for easier use and display.
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()

# Display the resulting DataFrame, which contains the average daily rate for City Hotels for each month.
city_hotel


# COMMAND ----------

# MAGIC %md
# MAGIC ###Merge the City and Hotel DataFrames, and rename the columns

# COMMAND ----------

# Merge the 'resort_hotel' and 'city_hotel' DataFrames based on the 'arrival_date_month' column.
# - The 'on' parameter specifies that the merging should be done using the 'arrival_date_month' column, 
#   which is common to both DataFrames.
# - This results in a DataFrame where each row contains the mean ADR (Average Daily Rate) for Resort Hotels and City Hotels for the same month.
final_hotel = resort_hotel.merge(city_hotel, on='arrival_date_month')

# Rename the columns in the resulting DataFrame to provide more descriptive names.
# - 'month' refers to the 'arrival_date_month' column, which contains the months of arrival.
# - 'price_for_resort' refers to the mean ADR for Resort Hotels for each month.
# - 'price_for_city_hotel' refers to the mean ADR for City Hotels for each month.
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']

# Display the merged DataFrame 'final_hotel'.
# This DataFrame now shows, for each month, the average daily rates for both Resort Hotels and City Hotels.
final_hotel

# COMMAND ----------

# MAGIC %md
# MAGIC ###We need to sort the months correctly

# COMMAND ----------

# Import the external module that provides functionality to sort DataFrames by months or weeks.
import sort_dataframeby_monthorweek as sd

# Define a function 'sort_month' that takes a DataFrame (df) and a column name as inputs.
# The purpose of this function is to sort the DataFrame based on the months in the specified column.
# - 'df' is the DataFrame that needs to be sorted.
# - 'column_name' is the name of the column that contains the month values you want to sort by.
# The function uses 'Sort_Dataframeby_Month' from the imported module 'sd' to sort the DataFrame by month.
def sort_month(df, column_name):
    return sd.Sort_Dataframeby_Month(df, column_name)


# COMMAND ----------

# Sort the 'final_hotel' DataFrame by the 'month' column using the previously defined 'sort_month' function.
# The 'sort_month' function sorts the DataFrame based on calendar months (e.g., January, February, etc.).
# - 'final_hotel' is the DataFrame containing hotel data, including information about each month.
# - 'month' is the column in 'final_hotel' that contains the names of the months, which will be sorted chronologically.
final_prices = sort_month(final_hotel, 'month')

# Display the sorted DataFrame.
# The DataFrame 'final_prices' is now sorted by month, ensuring the rows are organized in the correct calendar order.
final_prices


# COMMAND ----------

# MAGIC %md
# MAGIC ###We can now plot the rate for both city and resort hotels for each month of the year

# COMMAND ----------

# Create a new figure with a specified size using Matplotlib.
# This will create a plot area with a width of 17 and height of 8 inches to accommodate the line chart.
plt.figure(figsize=(17, 8))

# Create a line plot using Plotly Express (px) to visualize the price trends for Resort Hotels and City Hotels over the months.
# - 'final_prices' is the DataFrame containing the monthly prices for both hotel types.
# - 'x' refers to the 'month' column, which is the x-axis (representing months).
# - 'y' is set to both 'price_for_resort' and 'price_for_city_hotel', meaning that the plot will display two lines:
#     - One for the average daily rate of Resort Hotels.
#     - Another for the average daily rate of City Hotels.
# - 'title' adds a title to the plot, in this case, indicating that the plot shows the room price trends over the months.
# - 'template' specifies the 'seaborn' theme,  for a light grey background and muted colors for a softer visual look..
px.line(final_prices, x='month', y=['price_for_resort', 'price_for_city_hotel'],
        title='Room price per night over the Months', template='seaborn')


# COMMAND ----------

# MAGIC %md
# MAGIC ##Question: Which are the busy months?

# COMMAND ----------

# MAGIC %md
# MAGIC ###Count the number of guest by month for Resort guests

# COMMAND ----------

# Count the number of guests arriving each month for Resort Hotels by counting occurrences in the 'arrival_date_month' column.
# - 'data_resort' is the DataFrame filtered to only include non-canceled bookings for Resort Hotels.
# - 'value_counts()' counts how many times each month appears in the 'arrival_date_month' column, which effectively gives the number of guests arriving in each month.
# - 'reset_index()' converts the resulting Series into a DataFrame with two columns: one for the month and one for the count of guests.
resort_guests = data_resort['arrival_date_month'].value_counts().reset_index()

# Rename the columns of the DataFrame to give them more descriptive names:
# - 'month' for the month of guest arrival.
# - 'no of guests' for the number of guests arriving in that month.
resort_guests.columns = ['month', 'no of guests']

# Display the resulting DataFrame, which now contains two columns:
# - 'month': the name of the month.
# - 'no of guests': the number of guests who arrived at the Resort Hotel in that month.
resort_guests


# COMMAND ----------

# MAGIC %md
# MAGIC ###Count the number of guest by month for City guests

# COMMAND ----------

# Count the number of guests arriving each month for City Hotels by counting occurrences in the 'arrival_date_month' column.
# - 'data_city' is the DataFrame filtered to only include non-canceled bookings for City Hotels.
# - 'value_counts()' counts how many times each month appears in the 'arrival_date_month' column, which effectively gives the number of guests arriving in each month.
# - 'reset_index()' converts the resulting Series into a DataFrame with two columns: one for the month and one for the count of guests.
city_guests = data_city['arrival_date_month'].value_counts().reset_index()

# Rename the columns of the DataFrame to give them more descriptive names:
# - 'month' for the month of guest arrival.
# - 'no of guests' for the number of guests arriving in that month.
city_guests.columns = ['month', 'no of guests']

# Display the resulting DataFrame, which now contains two columns:
# - 'month': the name of the month.
# - 'no of guests': the number of guests who arrived at the City Hotel in that month.
city_guests


# COMMAND ----------

# MAGIC %md
# MAGIC Merge both DataFrames and rename the columns

# COMMAND ----------

# Merge the 'resort_guests' and 'city_guests' DataFrames on the 'month' column to combine the number of guests for both types of hotels.
# - The 'on' parameter specifies that the merging should be done using the 'month' column, which is common to both DataFrames.
# - This results in a DataFrame where each row contains the number of guests in both Resort Hotels and City Hotels for the same month.
final_guests = resort_guests.merge(city_guests, on='month')

# Rename the columns of the resulting DataFrame to provide more descriptive names:
# - 'month' represents the name of the month.
# - 'no of guests in resort' represents the number of guests in Resort Hotels for each month.
# - 'no of guests in city hotel' represents the number of guests in City Hotels for each month.
final_guests.columns = ['month', 'no of guests in resort', 'no of guests in city hotel']

# Display the merged DataFrame.
# This DataFrame now contains, for each month, the number of guests who stayed in both Resort Hotels and City Hotels.
final_guests


# COMMAND ----------

# MAGIC %md
# MAGIC ###Sort by Month

# COMMAND ----------

# Sort by month
final_guests = sort_month(final_guests,'month')
final_guests

# COMMAND ----------

# MAGIC %md
# MAGIC ###Produce the plot

# COMMAND ----------

# Create a line plot using Plotly Express (px) to visualize the number of guests in Resort Hotels and City Hotels across different months.
# - 'final_guests' is the DataFrame that contains the guest numbers for both hotel types, by month.
# Melt the 'final_guests' DataFrame from wide format to long format.
# - 'id_vars='month'' specifies that the 'month' column will remain unchanged (it will serve as the identifier variable).
# - 'value_vars=['no of guests in resort', 'no of guests in city hotel']' specifies the columns that will be unpivoted.
#   These columns will be combined into a new column ('no of guests'), with the original column names becoming values in the 'hotel_type' column.
# - 'var_name='hotel_type'' gives the name to the new column that stores the hotel types ('no of guests in resort' and 'no of guests in city hotel').
# - 'value_name='no of guests'' gives the name to the new column that will contain the values from the 'no of guests in resort' and 'no of guests in city hotel' columns.
final_guests_melted = final_guests.melt(id_vars='month', 
                                        value_vars=['no of guests in resort', 'no of guests in city hotel'], 
                                        var_name='hotel_type', value_name='no of guests')

# Create a line plot using Plotly Express (px) to visualize the number of guests in Resort Hotels and City Hotels over the months.
# - 'final_guests_melted' is the DataFrame in long format, making it easier to plot multiple series.
# - 'x='month'' sets the 'month' column as the x-axis, representing the months.
# - 'y='no of guests'' sets the 'no of guests' column as the y-axis, showing the number of guests.
# - 'color='hotel_type'' ensures that separate lines are plotted for Resort Hotels and City Hotels based on the values in the 'hotel_type' column.
# - 'title='Total no of guests per Month'' adds a title to the plot to describe the content of the visualization.
# - 'template='seaborn'' applies the 'seaborn' template, which provides a clean and minimalist styling similar to the Seaborn library in Python.
px.line(final_guests_melted, x='month', y='no of guests', color='hotel_type',
        title='Total no of guests per Month', template='seaborn')


# COMMAND ----------

# MAGIC %md
# MAGIC ##Question: How long do people stay at the hotels?

# COMMAND ----------

filter = df['is_canceled'] == 0
data = df[filter]
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a new column in the DataFrame called 'total_nights'

# COMMAND ----------

# Create a new column 'total_nights' in the 'data' DataFrame.
# This new column calculates the total number of nights stayed by a guest, which is the sum of:
# - 'stays_in_weekend_nights': The number of weekend nights the guest stayed.
# - 'stays_in_week_nights': The number of weeknights the guest stayed.
# Adding these two columns gives the total number of nights for each guest's stay.
data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

# Display the first five rows of the DataFrame to verify that the 'total_nights' column has been added correctly.
# 'head()' shows the first 5 rows, allowing a quick inspection of the data.
data.head()


# COMMAND ----------

# Group the 'data' DataFrame by 'total_nights' and 'hotel', and aggregate the results.
# - 'total_nights' represents the total number of nights stayed by a guest (calculated earlier).
# - 'hotel' indicates the type of hotel (e.g., Resort or City Hotel).
# - 'agg('count')' counts the number of occurrences for each combination of 'total_nights' and 'hotel'.
# - 'reset_index()' converts the result back into a DataFrame with default indexing.
stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()

# Select only the first three columns from the resulting DataFrame.
# - 'iloc[:, :3]' keeps the first three columns, which typically would be 'total_nights', 'hotel', and one of the aggregated columns (like 'is_canceled').
stay = stay.iloc[:, :3]

# Rename the 'is_canceled' column to 'Number of stays' to better reflect its content.
# - The count of 'is_canceled' is used to represent the total number of stays, as each row corresponds to a booking.
stay = stay.rename(columns={'is_canceled': 'Number of stays'})

# Display the resulting DataFrame.
# This DataFrame shows the total number of stays for different combinations of 'total_nights' and 'hotel' type.
stay


# COMMAND ----------

# Create a grouped bar chart using Plotly Express (px) to visualize the number of stays for different 'total_nights' values and hotel types.
# - 'data_frame=stay' specifies that the 'stay' DataFrame is the source of the data for the chart.
# - 'x='total_nights'' sets the x-axis to represent the 'total_nights' column, showing the total number of nights guests stayed.
# - 'y='Number of stays'' sets the y-axis to represent the 'Number of stays', displaying the total number of stays for each 'total_nights' value.
# - 'color='hotel'' ensures that the bars are color-coded by hotel type, allowing you to distinguish between Resort Hotels and City Hotels.
# - 'barmode='group'' specifies that the bars for each hotel type should be grouped side by side, rather than stacked.
# - 'template='seaborn'' applies the Seaborn-style template for a cleaner, minimalist look and feel for the chart.
px.bar(data_frame=stay, x='total_nights', y='Number of stays', color='hotel', barmode='group',
       template='seaborn')


# COMMAND ----------

# MAGIC %md
# MAGIC #Assumptions Validations / Rejections

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cancelation Rates at city hotels are higher than resort hotels
# MAGIC The type of hotel decides the cancelation rate with higher cancellations in city hotels as compared to resort hotels due to variety of facilities available in resort hotels.

# COMMAND ----------

# MAGIC %md
# MAGIC ###First, look at the percentage cancellations

# COMMAND ----------

# Calculate the total number of canceled bookings in the DataFrame.
# - 'df[df['is_canceled'] == 1]' filters the DataFrame to include only rows where bookings were canceled (is_canceled == 1).
# - 'len()' counts the total number of rows in this filtered DataFrame, giving the total number of cancellations.
is_cancelled = len(df[df['is_canceled'] == 1])

# Calculate the percentage of cancellations.
# - 'len(df)' gives the total number of bookings in the original DataFrame.
# - The cancellation percentage is calculated by dividing the total number of cancellations by the total number of bookings, and then multiplying by 100 to convert it to a percentage.
percentage_cancelations = is_cancelled / len(df) * 100

# Print the total percentage of cancellations with a formatted string for easy readability.
print(f"Total percentage of cancellations: {percentage_cancelations:.2f}%")

# Calculate the distribution of reservation statuses as a percentage.
# - 'df['reservation_status'].value_counts(normalize=True)' counts the occurrences of each unique value in the 'reservation_status' column.
# - 'normalize=True' means the counts are normalized, or divided by the total number of rows, giving a proportion.
# - Multiplying by 100 converts the proportions to percentages.
df['reservation_status'].value_counts(normalize=True) * 100


# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a Pearon correlation matrix over the 'is_canceled' column

# COMMAND ----------

# Calculate the correlation matrix for the DataFrame 'df' using the Pearson correlation method.
# - 'df.corr(method="pearson", numeric_only=True)' calculates the correlation matrix only for numeric columns.
#   Pearson correlation measures the linear relationship between pairs of variables, returning values between -1 and 1.
# - 'numeric_only=True' ensures that only numeric columns are included in the calculation, avoiding errors with non-numeric data.
# - 'is_canceled' is the target column, and the correlation values with this column will show how strongly each numeric variable in the DataFrame correlates with the booking cancellation status.
correlation_matrix = df.corr(method="pearson", numeric_only=True)['is_canceled'][:]

# Display the correlation values for each variable in relation to the 'is_canceled' column.
# The resulting series will show how strongly each variable is correlated with booking cancellations.
correlation_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC THe highest positive correlation is 'lead_time', the highest negative correlation is 'total_of_special requests'

# COMMAND ----------

# Create the count plot for hotels and cancellations.
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='hotel', hue='is_canceled')
plt.title('Hotel Cancellations by Type')
plt.xlabel('Hotel Type')
plt.ylabel('Number of Bookings')
plt.legend(title='Is Canceled', labels=['Not Canceled', 'Canceled'])

# Calculate the number of cancellations in both hotel types.
cancelation_in_resort_hotel = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 1)]
cancelation_in_city_hotel = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 1)]

# Calculate the percentage of cancellations for each hotel type.
percentage_cancelations_in_resort_hotel = len(cancelation_in_resort_hotel) / len(df[df['hotel'] == 'Resort Hotel']) * 100
percentage_cancelations_in_city_hotel = len(cancelation_in_city_hotel) / len(df[df['hotel'] == 'City Hotel']) * 100

# Display percentages as text below the plot.
textstr = (f"Percentage of cancellations in Resort Hotel: {percentage_cancelations_in_resort_hotel:.2f}%\n"
           f"Percentage of cancellations in City Hotel: {percentage_cancelations_in_city_hotel:.2f}%")

# Place the text below the plot
plt.figtext(0.5, -0.05, textstr, wrap=True, horizontalalignment='center', fontsize=12)

# Adjust layout to make room for the text below the chart.
plt.tight_layout()

# Show the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###This validates our assumption that cancelation rates at city hotels are higher!

# COMMAND ----------

# MAGIC %md
# MAGIC ##The earlier the booking made, higher the chances of cancellation.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a plot that shows cancellations versus lead time

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the FacetGrid, faceting by 'is_canceled'
grid = sns.FacetGrid(df, col='is_canceled', height=5, aspect=1.5)

# Map histograms of 'lead_time' onto the grid, with bins and color customization
grid.map(plt.hist, 'lead_time', bins=30, color='skyblue', edgecolor='black')

# Add titles and labels for better context
grid.set_titles(col_template="{col_name} Canceled")
grid.set_axis_labels('Lead Time (days)', 'Frequency')

# Add a legend
grid.add_legend()

# Adjust the layout for better fit
plt.tight_layout()

# Show the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Here, we see that the longer the lead time is, the lower the cancellations are, therefore this assumption is invalid!

# COMMAND ----------

# MAGIC %md
# MAGIC ##This assumption has been invalidated

# COMMAND ----------

# MAGIC %md
# MAGIC ##The more children and babies, the higher the chance of cancelation

# COMMAND ----------

# Filter the DataFrame to include only rows where there are either children or babies.
# - 'df['children'] != 0' checks for rows where the 'children' column is not equal to 0 (i.e., children are present).
# - 'df['babies'] != 0' checks for rows where the 'babies' column is not equal to 0 (i.e., babies are present).
# - The '|' operator is a logical OR, so we select rows where either of these conditions is true (i.e., either children or babies are present).
filtered_df = df.loc[(df['children'] != 0) | (df['babies'] != 0)]

# Calculate the total number of rows in the filtered DataFrame, i.e., how many bookings involve children or babies.
num_bookings_with_children_or_babies = len(filtered_df)

# Calculate the total number of rows in the original DataFrame, i.e., the total number of bookings.
total_bookings = len(df)

# Calculate the percentage of bookings that involve either children or babies.
# This is done by dividing the number of bookings with children or babies by the total number of bookings.
percentage_of_bookings_with_children_or_babies = (num_bookings_with_children_or_babies / total_bookings) * 100

# Print or return the percentage value.
percentage_of_bookings_with_children_or_babies


# COMMAND ----------

# MAGIC %md
# MAGIC The number of customers having children or babies or both are only 8% of the total population. Therefore this information can be ignored as it will not play a significant role in deciding whether to cancel the booking or not. Assumption 3 is therefore inconclusive

# COMMAND ----------

# MAGIC %md
# MAGIC ###This assumption is inconclusive 

# COMMAND ----------

# MAGIC %md
# MAGIC ##A repeated guest is less likely to cancel current booking.

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Plot a countplot to visualize the distribution of repeated vs. new guests with respect to cancellations.
# - 'x='is_repeated_guest'' plots the guest type on the x-axis (whether they are new or returning).
# - 'hue='is_canceled'' splits the bars based on cancellation status (whether the booking was canceled or not).
plt.figure(figsize=(8, 6))  # Adjust plot size for readability
sns.countplot(data=df, x='is_repeated_guest', hue='is_canceled')
plt.title('Cancellation Distribution by Guest Type')
plt.xlabel('Guest Type (0 = New Guest, 1 = Returning Guest)')
plt.ylabel('Number of Bookings')
plt.legend(title='Is Canceled', labels=['Not Canceled', 'Canceled'])
plt.show()

# Filter the DataFrame for cancellations made by new guests (is_repeated_guest == 0) and returning guests (is_repeated_guest == 1).
# - 'new_guest' filters for new guests who canceled (is_repeated_guest == 0 and is_canceled == 1).
# - 'old_guest' filters for returning guests who canceled (is_repeated_guest == 1 and is_canceled == 1).
new_guest = df[(df['is_repeated_guest'] == 0) & (df['is_canceled'] == 1)]
old_guest = df[(df['is_repeated_guest'] == 1) & (df['is_canceled'] == 1)]

# Calculate the percentage of cancellations among new guests.
# - 'len(new_guest)' gives the number of cancellations among new guests.
# - 'len(df[df['is_repeated_guest'] == 0])' gives the total number of new guest bookings.
percentage_new_guest_cancelations = (len(new_guest) / len(df[df['is_repeated_guest'] == 0])) * 100

# Calculate the percentage of cancellations among returning guests.
# - 'len(old_guest)' gives the number of cancellations among returning guests.
# - 'len(df[df['is_repeated_guest'] == 1])' gives the total number of returning guest bookings.
percentage_old_guest_cancelations = (len(old_guest) / len(df[df['is_repeated_guest'] == 1])) * 100

# Display the calculated percentages with improved formatting.
print(f"Percentage of cancellations among new guests: {percentage_new_guest_cancelations:.2f}%")
print(f"Percentage of cancellations among returning guests: {percentage_old_guest_cancelations:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC As seen in the correlation table, the above graph bolsters the evidence that maximum customers are new comers and they are less likely to cancel their current booking. Old guests are less likely to cancel the booking (14%). Therefore this assumption holds true.

# COMMAND ----------

# MAGIC %md
# MAGIC ###This assumption holds true!

# COMMAND ----------

# MAGIC %md
# MAGIC ##Higher previous cancellations lead to cancellation of current bookings

# COMMAND ----------

# MAGIC %md
# MAGIC ###Plot the previous cancellations against 'is_canceled'

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Create a countplot to visualize the relationship between previous cancellations and current booking cancellations.
# - 'x='previous_cancellations'' displays the number of previous cancellations on the x-axis.
# - 'hue='is_canceled'' shows whether the current booking was canceled (colored by cancellation status).
plt.figure(figsize=(8, 6))  # Adjust the plot size for better readability.
sns.countplot(data=df, x='previous_cancellations', hue='is_canceled', palette='Set2')

# Add a title to the plot for better context.
plt.title('Current Cancellations Based on Previous Cancellations')

# Label the axes for clarity.
plt.xlabel('Number of Previous Cancellations')
plt.ylabel('Number of Bookings')

# Customize the legend for clarity.
plt.legend(title='Current Booking Canceled', labels=['Not Canceled', 'Canceled'])

# Display the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Maximum customers have 0 previous cancellations. They are less likely to cancel the current booking. However, customers who have cancelled once earlier are more likely to cancel the current booking. This also matches with the positive correlation between previous_cancellations and is_cancelled. This assumption holds true.

# COMMAND ----------

# MAGIC %md
# MAGIC ###This assumption has been validated

# COMMAND ----------

# MAGIC %md
# MAGIC ##If room assigned is not the reserved room type, customer might cancel.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculate number of cancellations when reserved room type does not equal assigned room type

# COMMAND ----------

# Filter the DataFrame to include only rows where the reserved room type 
# does not match the assigned room type. This identifies bookings where 
# guests were not given the room type they initially reserved.
temp = df.loc[df['reserved_room_type'] != df['assigned_room_type']]

# Calculate the percentage of cancellations among bookings where the reserved
# room type was different from the assigned room type. 
# - 'is_canceled' column contains 1 if the booking was canceled, and 0 if not.
# - 'value_counts(normalize=True)' counts the occurrences of cancellations (1) and non-cancellations (0),
#   then normalizes the counts by dividing by the total number of rows to get proportions.
# - Multiplying by 100 converts the proportions to percentages.
cancelation_percentage = temp['is_canceled'].value_counts(normalize=True) * 100

# Display the percentage of cancellations and non-cancellations
cancelation_percentage


# COMMAND ----------

# MAGIC %md
# MAGIC ###Add a plot to visualize

# COMMAND ----------

# Filter the data where the reserved room type does not match the assigned room type
temp = df.loc[df['reserved_room_type'] != df['assigned_room_type']]

# Calculate the percentage of cancellations vs non-cancellations for mismatched room types
cancelation_percentage = temp['is_canceled'].value_counts(normalize=True) * 100

# Create a bar plot to visualize the percentages of cancellations and non-cancellations
plt.figure(figsize=(8, 6))
sns.barplot(x=cancelation_percentage.index, y=cancelation_percentage.values, palette="Set2")

# Add titles and labels to make the plot informative
plt.title('Percentage of Cancellations for Mismatched Room Assignments')
plt.xlabel('Is Canceled (0 = Not Canceled, 1 = Canceled)')
plt.ylabel('Percentage')

# Display the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC The assumption that there more cancellations when assigned room type is different from reserved room type is not valid. There are only 5% cancellations in such a case.

# COMMAND ----------

# MAGIC %md
# MAGIC ###This assumption does not hold true

# COMMAND ----------

# MAGIC %md
# MAGIC ##If  number of booking changes made is high, chance of cancellation is low

# COMMAND ----------

# Set Seaborn style for better aesthetics
sns.set_style("whitegrid")

# Create the point plot
plt.figure(figsize=(8, 6))
sns.pointplot(data=df, x='booking_changes', y='is_canceled', color='blue', markers="o", linestyles='-', scale=0.7)

# Add titles and labels to make the plot informative
plt.title('Effect of Booking Changes on Cancellations', fontsize=14)
plt.xlabel('Number of Booking Changes', fontsize=12)
plt.ylabel('Cancellation Rate (Proportion of Cancellations)', fontsize=12)

# Improve the visibility of axis ticks and labels
plt.xticks(ticks=range(df['booking_changes'].nunique()), fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout to avoid cutting off elements
plt.tight_layout()

# Display the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Assumption 8 about the bookings does not hold as there is no trend in it's impact on the cancellation of bookings.

# COMMAND ----------

# MAGIC %md
# MAGIC ##This assumption has been invalidated

# COMMAND ----------

# MAGIC %md
# MAGIC ##Refundable bookings or those without deposits have higher cancellations.

# COMMAND ----------

# MAGIC %md
# MAGIC Plot 'deposit_type' against cancellations

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for better aesthetics
sns.set_style("whitegrid")

# Adjust figure size for better readability
plt.figure(figsize=(8, 6))

# Create the count plot with additional customization
sns.countplot(x="deposit_type", hue="is_canceled", data=df, palette="Set2")

# Add title and labels for clarity
plt.title('Cancellations by Deposit Type', fontsize=14)
plt.xlabel('Deposit Type', fontsize=12)
plt.ylabel('Number of Bookings', fontsize=12)

# Customize the legend for clarity
plt.legend(title='Is Canceled', labels=['Not Canceled', 'Canceled'], loc='upper right')

# Improve the visibility of axis ticks and labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Contrary to our assumptions, bookings that are non_refundable are canceled.

# COMMAND ----------

# MAGIC %md
# MAGIC ##This assumption does not hold True

# COMMAND ----------

# MAGIC %md
# MAGIC ##Assumption: If the # of days in waiting list is high, cancelations are higher

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Set the Seaborn style for better aesthetics
sns.set_style("whitegrid")

# Create the relational plot (relplot) with line representation
# - 'x='days_in_waiting_list'' sets the x-axis to the number of days in the waiting list.
# - 'y='is_canceled'' plots the y-axis showing whether a booking was canceled (1) or not (0).
# - 'kind='line'' creates a line plot.
# - 'estimator=None' ensures that the plot shows each point as is without any aggregation.
plt.figure(figsize=(10, 6))
sns.relplot(data=df, x='days_in_waiting_list', y='is_canceled', kind='line', estimator=None, color='blue', marker='o')

# Add title and axis labels for clarity
plt.title('Effect of Waiting List Days on Cancellations', fontsize=14)
plt.xlabel('Days in Waiting List', fontsize=12)
plt.ylabel('Cancellation Status (0 = Not Canceled, 1 = Canceled)', fontsize=12)

# Adjust layout to ensure everything fits properly
plt.tight_layout()

# Show the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC No relation can be established between days_in_waiting_list and is_canceled. Therefore, we will take this feature for further analysis. Therefore, this assumption can be discarded.

# COMMAND ----------

# MAGIC %md
# MAGIC ###This assumption does not hold true

# COMMAND ----------

# MAGIC %md
# MAGIC #End of Notebook
