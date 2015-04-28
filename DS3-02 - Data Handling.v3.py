
# coding: utf-8

# # DS3 Data Handling

# ## R. Burke Squires

# ### NIAID Bioinformatics and Computational Biosciences Branch (BCBB)

# ---

# # Outline:

# - Intro to Python
#     - Learn python in Y minutes
# 
# - Importing Data
#     - csv import
#     - Excel import
#     - Database import
#     - Web import
# 
# - Pandas
#     - Importing Data
#     - Removing missing values
#     - Fun with Columns
#     - Filtering
#     - Grouping
#     - Plotting
#     - Getting data out
#     - Reading and writing to Excel

# ---

# ### Learn Python in Y Minutes

# See Learn Python in Y Minutes IPython Notebook

# ---

# # An Introduction to Pandas
# 
# ** Presentation originally developed by Michael Hansen, modified slightly by Jeff Shelton **
# 
# **pandas** is a Python package providing fast, flexible, and expressive data structures designed to work with *relational* or *labeled* data both. It is a fundamental high-level building block for doing practical, real world data analysis in Python. 
# 
# pandas is well suited for:
# 
# - Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet
# - Ordered and unordered (not necessarily fixed-frequency) time series data.
# - Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels
# - Any other form of observational / statistical data sets. The data actually need not be labeled at all to be placed into a pandas data structure
# 
# 
# Key features:
#     
# - Easy handling of **missing data**
# - **Size mutability**: columns can be inserted and deleted from DataFrame and higher dimensional objects
# - Automatic and explicit **data alignment**: objects can be explicitly aligned to a set of labels, or the data can be aligned automatically
# - Powerful, flexible **group by functionality** to perform split-apply-combine operations on data sets
# - Intelligent label-based **slicing, fancy indexing, and subsetting** of large data sets
# - Intuitive **merging and joining** data sets
# - Flexible **reshaping and pivoting** of data sets
# - **Hierarchical labeling** of axes
# - Robust **IO tools** for loading data from flat files, Excel files, databases, and HDF5
# - **Time series functionality**: date range generation and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging, etc.

# In[4]:

import pandas


# ## Data Import

# ### Import CVS

# Next, let's read in [our data](data/weather_year.csv).
# Because it's in a CSV file, we can use pandas' `read_csv` function to pull it directly into a [DataFrame](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe).

# In[7]:

help(pandas.read_csv, sep="\t")


# In[2]:

data = pandas.read_csv("data/weather_year.csv")


# We can get a summary of the DataFrame by asking for some information:

# In[3]:

data.info()


# ### Import Excel

# Using `len` on a DataFrame will give you the number of rows. You can get the column names using the `columns` property.
# 
# The read_excel() method can read Excel 2003 (.xls) and Excel 2007 (.xlsx) files using the xlrd Python module and use the same parsing code as the above to convert tabular data into a DataFrame. See the cookbook for some advanced strategies
# 
# Besides read_excel you can also read Excel files using the ExcelFile class.

# In[8]:

data = pandas.read_excel("data/weather_year.xlsx")
data.info()


# In[ ]:

data = pandas.read_excel("/Users/squiresrb/Dropbox/NIEHS/")


# In[ ]:

# using the ExcelFile class
xls = pandas.ExcelFile('data/weather_year.xlsx')
xls.parse('Sheet1', index_col=None, na_values=['NA'])


# In[9]:

# using the read_excel function
data = pandas.read_excel('data/weather_year.xlsx', 'Sheet1', index_col=None, na_values=['NA'])


# In[10]:

#Using the sheet index:
data = pandas.read_excel('data/weather_year.xlsx', 0, index_col=None, na_values=['NA'])


# In[11]:

#Using all default values:
data = pandas.read_excel('data/weather_year.xlsx')


# New in version 0.16.
# 
# read_excel can read more than one sheet, by setting sheetname to either a list of sheet names, a list of sheet positions, or None to read all sheets.

# ### Import Form a Database (SQL)

# First we are going to create a sqlite3 database using hte columns and data from the csv file above. Then we will connect to the databae and read from it.

# In[12]:

import pandas as pd
import sqlite3

weather_df = pd.read_csv("data/weather_year.csv")
con = sqlite3.connect("data/test_db.sqlite")
con.execute("DROP TABLE IF EXISTS weather_year")
pd.io.sql.to_sql(weather_df, "weather_year", con)


# In[13]:

con = sqlite3.connect("data/test_db.sqlite")
data = pandas.read_sql("SELECT * from weather_year", con)
data.info()


# ### Additional Formats

# JSON: JavaScript Object Notation) is a lightweight data-interchange format
# HDF: high performance HDF5 format using the excellent 

# ---

# ## Getting Started

# OK, let's get started by importing the pandas library.

# In[14]:

import pandas


# Next, let's read in our data.
# Because it's in a CSV file, we can use pandas' `read_csv` function to pull it directly into a [DataFrame](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe).

# In[15]:

data = pandas.read_csv("data/weather_year.csv")


# We can get a summary of the DataFrame by asking for some information:

# In[16]:

data.info()


# This gives us a lot of insight. First, we can see that there are 819 rows (entries). 
# 
# Each column is printed along with however many "non-null" values are present.
# 
# Lastly, the data types (dtypes) of the columns are printed at the very bottom. We can see that there are 10 `float64`, 3 `int64`, and 5 `object` columns.

# In[17]:

len(data)


# Using `len` on a DataFrame will give you the number of rows. You can get the column names using the `columns` property.

# In[18]:

data.columns


# Columns can be accessed in two ways. The first is using the DataFrame like a dictionary with string keys:

# In[19]:

data["EDT"]


# You can get multiple columns out at the same time by passing in a list of strings.

# In[22]:

data[['EDT', 'Mean TemperatureF']]


# The second way to access columns is using the dot syntax. This only works if your column name could also be a Python variable name (i.e., no spaces), and if it doesn't collide with another DataFrame property or function name (e.g., count, sum).

# In[ ]:

data.


# We'll be mostly using the dot syntax here because you can auto-complete the names in IPython. The first pandas function we'll learn about is `head()`. This gives us the first 5 items in a column (or the first 5 rows in the DataFrame).

# In[23]:

data.EDT.head()


# Passing in a number `n` gives us the first `n` items in the column. There is also a corresponding `tail()` method that gives the *last* `n` items or rows.

# In[24]:

data.EDT.head(10)


# This also works with the dictionary syntax.

# In[25]:

data["EDT"].head()


# In[26]:

data["EDT"].describe()


# ## Fun with Columns

# The column names in `data` are a little unwieldy, so we're going to rename them. This is as easy as assigning a new list of column names to the `columns` property of the DataFrame.
# 
# aeid = assay endpoint id (unique id)
# assay_component_endpoint_name = name of assay endpoint
# analysis_direction = the analyzed positive (upward) or negative (downward) direction 
# signal_direction = the direction observed of the detected signal 
# normalized_data_type = fold induction or percent positive control 
# key_positive_control = positive control used to normalize data
# zprm.mdn = z-prime median across all plates (where applicable)
# zprm.mad = z prime median absolute deviation (mad)
# ssmd.mdn = strictly standardized mean difference median across all plates
# ssmd.mad = strictly standardized mean difference mad across all plates
# cv.mdn = coefficient of variation median across all plates
# cv.mad = coefficient of variation mad across all plates
# sn.mdn = signal-to-noise median across all plates
# sn.mad = signal-to-noise mad across all plates
# sb.mdn = signal-to-background median across all plates
# sb.mad = signal-to-background mad across all plates

# In[27]:

data.columns = ["date", "max_temp", "mean_temp", "min_temp", "max_dew",
                "mean_dew", "min_dew", "max_humidity", "mean_humidity",
                "min_humidity", "max_pressure", "mean_pressure",
                "min_pressure", "max_visibilty", "mean_visibility",
                "min_visibility", "max_wind", "mean_wind", "min_wind",
                "precipitation", "cloud_cover", "events", "wind_dir"]


# These should be in the same order as the original columns. Let's take another look at our DataFrame summary. We can see that the second column is not entitled 'assay_endpoint_id' instead of 'aeid'.

# In[28]:

data.info()


# In[ ]:

data.pre


# Now our columns can all be accessed using the dot syntax!

# In[ ]:

data.mean_temp.head()


# There are lots useful methods on columns, such as `std()` to get the standard deviation. Most of pandas' methods will happily ignore missing values like `NaN`.

# In[ ]:

data.mean_temp.std()


# If you want to add labels and save the plot as a `png` file that is sized 800 pixels by 600 pixels:

# By the way, many of the column-specific methods also work on the entire DataFrame. Instead of a single number, you'll get a result for each column.

# In[ ]:

data.std()


# In[ ]:

data.mean_temp.max()


# ## Data Transformations

# Methods like `sum()` and `std()` work on entire columns. We can run our own functions across all values in a column (or row) using `apply()`.
# 
# To give you an idea of how this works, let's consider the "date" column in our DataFrame (formally "EDT").

# In[29]:

data.date.head()


# We can use the `values` property of the column to get a list of values for the column. Inspecting the first value reveals that these are strings with a particular format.

# In[30]:

first_date = data.date.values[0]
first_date


# The `strptime` function from the `datetime` module will make quick work of this date string. There are many [more shortcuts available](http://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior) for `strptime`.

# In[31]:

# Import the datetime class from the datetime module
from datetime import datetime

# Convert date string to datetime object
datetime.strptime(first_date, "%Y-%m-%d")


# Using the `apply()` method, which takes a function (**without** the parentheses), we can apply `strptime` to each value in the column. We'll overwrite the string date values with their Python `datetime` equivalents.

# In[32]:

# Define a function to convert strings to dates
def string_to_date(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d")

# Run the function on every date string and overwrite the column
data.date = data.date.apply(string_to_date)
data.date.head()


# Let's go one step futher. Each row in our DataFrame represents the weather from a single day. Each row in a DataFrame is associated with an *index*, which is a label that uniquely identifies a row.
# 
# Our row indices up to now have been auto-generated by pandas, and are simply integers from 0 to 365. If we use dates instead of integers for our index, we will get some extra benefits from pandas when plotting later on. Overwriting the index is as easy as assigning to the `index` property of the DataFrame.

# In[33]:

data.index = data.date
data.info()


# Now we can quickly look up a row by its date with the `loc[]` property \[[see docs](http://pandas.pydata.org/pandas-docs/stable/indexing.html)], which locates records by label.

# In[34]:

data.loc[datetime(2012, 8, 19)]


# We can also access a row (or range of rows) with the `iloc[]` property, which locates records by integer index.

# In[35]:

data.max_temp.iloc[7:15]


# With all of the dates in the index now, we no longer need the "date" column. Let's drop it.

# In[36]:

data = data.drop("CREATE_DATE", axis=1)
data.columns


# Note that we need to pass in `axis=1` in order to drop a column. For more details, check out the [documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html) for `drop`. The index values can now be accessed as `data.index.values`.

# In[37]:

data = data.drop("date", axis=1)
data.columns


# ## Handing Missing Values

# Pandas considers values like `NaN` and `None` to represent missing data. The `count()` function [[see docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.count.html)] can be used to tell whether values are missing. We use the parameter `axis=0` to indicate that we want to perform the count by rows, rather than columns.

# In[38]:

data.count(axis=0)


# It is pretty obvious that there are a lot of `NaN` entrys for the `events` column; 204 to be exact. Let's take a look at a few values from the `events` column:

# In[39]:

data.events.head(10)


# This isn't exactly what we want. One option is to drop all rows in the DataFrame with missing "events" values using the `dropna()` function \[[see docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)].

# In[40]:

data.dropna(subset=["events"]).info()


# Note that this didn't affect `data`; we're just looking at a copy.
# 
# Instead of dropping rows with missing values, let's fill them with empty strings (you'll see why in a moment). This is easily done with the `fillna()` function. We'll go ahead and overwrite the "events" column with empty string missing values instead of `NaN`.

# In[41]:

data.events = data.events.fillna("")
data.events.head(10)


# Now we repeat the `count` function for the `events` column:

# In[42]:

data.events.count()


# As desired, there are no longer any empty entries in the `events` column. Why did we not need the `axis=0` parameter this time?

# ## Iteratively Accessing Rows

# You can iterate over each row in the DataFrame with `iterrows()`. Note that this function returns **both** the index and the row. Also, you must access columns in the row you get back from `iterrows()` with the dictionary syntax.

# In[43]:

num_rain = 0
for idx, row in data.iterrows():
    if "Rain" in row["events"]:
        num_rain += 1

"Days with rain: {0}".format(num_rain)


# ## Filtering

# Most of your time using pandas will likely be devoted to selecting rows of interest from a DataFrame. In addition to strings, the dictionary syntax accepts requests like:

# In[45]:

freezing_days = data[data.max_temp <= 32]
freezing_days.info()


# We get back another DataFrame with fewer rows (21 in this case). This DataFrame can be filtered down even more by adding a constrain that the temperature be greater than 20 degrees, in addition to being below freezing.

# In[ ]:

cold_days = freezing_days[freezing_days.min_temp >= 20]
cold_days.info()


# To see the high and low temperatures for the selected days:

# In[ ]:

cold_days[["max_temp","min_temp"]]


# Using boolean operations, we could have chosen to apply both filters to the original DataFrame at the same time.

# In[46]:

data[(data.max_temp <= 32) & (data.min_temp >= 20)]


# It's important to understand what's really going on underneath with filtering. Let's look at what kind of object we actually get back when creating a filter.

# In[ ]:

temp_max = data.max_temp <= 32
type(temp_max)


# This is a pandas `Series` object, which is the one-dimensional equivalent of a DataFrame. Because our DataFrame uses datetime objects for the index, we have a specialized `TimeSeries` object.
# 
# What's inside the filter?

# In[ ]:

temp_max


# Our filter is nothing more than a `Series` with a *boolean value for every item in the index*. When we "run the filter" as so:

# In[ ]:

data[temp_max].info()


# pandas lines up the rows of the DataFrame and the filter using the index, and then keeps the rows with a `True` filter value. That's it.
# 
# Let's create another filter.

# In[ ]:

temp_min = data.min_temp >= 20
temp_min


# Now we can see what the boolean operations are doing. Something like `&` (**not** `and`)...

# In[ ]:

temp_min & temp_max


# ...is just lining up the two filters using the index, performing a boolean AND operation, and returning the result as another `Series`.
# 
# We can do other boolean operations too, like OR:

# In[ ]:

temp_min | temp_max


# Because the result is just another `Series`, we have all of the regular pandas functions at our disposal. The `any()` function returns `True` if any value in the `Series` is `True`.

# In[ ]:

temp_both = temp_min & temp_max
temp_both.any()


# Sometimes filters aren't so intuitive. This (sadly) doesn't work:

# In[ ]:

try:
    data["Rain" in data.events]
except:
    pass # "KeyError: no item named False"


# We can wrap it up in an `apply()` call fairly easily, though:

# In[ ]:

data[data.events.apply(lambda e: "Rain" in e)].info()


# We'll replace "T" with a very small number, and convert the rest of the strings to floats:

# In[47]:

# Convert precipitation to floating point number
# "T" means "trace of precipitation"
def precipitation_to_float(precip_str):
    if precip_str == "T":
        return 1e-10  # Very small value
    return float(precip_str)

data.precipitation = data.precipitation.apply(precipitation_to_float)
data.precipitation.head()


# ---

# ## Ordering: Sorting data

# Sort by the events column, ascending

# In[62]:

get_ipython().magic('pinfo data.sort')


# In[64]:

data.sort(['max_temp', 'mean_temp'])


# In[ ]:




# ---

# # Data Transformation

# ---

# ## Grouping

# Besides `apply()`, another great DataFrame function is `groupby()`.
# It will group a DataFrame by one or more columns, and let you iterate through each group.
# 
# As an example, let's group our DataFrame by the "cloud_cover" column (a value ranging from 0 to 8).

# In[49]:

cover_temps = {}
for cover, cover_data in data.groupby("cloud_cover"):
    cover_temps[cover] = cover_data.mean_temp.mean()  # The mean mean temp!
cover_temps


# When you iterate through the result of `groupby()`, you will get a tuple.
# The first item is the column value, and the second item is a filtered DataFrame (where the column equals the first tuple value).
# 
# You can group by more than one column as well.
# In this case, the first tuple item returned by `groupby()` will itself be a tuple with the value of each column.

# In[50]:

for (cover, events), group_data in data.groupby(["cloud_cover", "events"]):
    print("Cover: {0}, Events: {1}, Count: {2}".format(cover, events, len(group_data)))


# ## Reshaping: Creating New Columns

# Weather events in our DataFrame are stored in strings like "Rain-Thunderstorm" to represent that it rained and there was a thunderstorm that day. Let's split them out into boolean "rain", "thunderstorm", etc. columns.
# 
# First, let's discover the different kinds of weather events we have with `unique()`.

# In[85]:

a= 1
print(a)
print(a+1)


# In[51]:

data.events.unique()


# Looks like we have "Rain", "Thunderstorm", "Fog", and "Snow" events. Creating a new column for each of these event kinds is a piece of cake with the dictionary syntax.

# In[52]:

for event_kind in ["Rain", "Thunderstorm", "Fog", "Snow"]:
    col_name = event_kind.lower()  # Turn "Rain" into "rain", etc.
    data[col_name] = data.events.apply(lambda e: event_kind in e)
data.info()


# Our new columns show up at the bottom. We can access them now with the dot syntax.

# In[53]:

data.rain


# We can also do cool things like find out how many `True` values there are (i.e., how many days had rain)...

# In[54]:

data.rain.sum()


# ...and get all the days that had both rain and snow!

# In[55]:

data[data.rain & data.snow].info()


# ## Getting Data Out

# Writing data out in pandas is as easy as getting data in. To save our DataFrame out to a new csv file, we can just do this:

# In[76]:

data.to_csv("data/weather-mod.csv")


# Want to make that tab separated instead? No problem.

# In[77]:

data.to_csv("data/weather-mod.tsv", sep="\t")


# There's also support for [reading and writing Excel files](http://pandas.pydata.org/pandas-docs/stable/io.html#excel-files), if you need it.

# ## Updating a Cell

# In[78]:

for idx, row in data.iterrows():
    data.max_temp.loc[idx] = 0
any(data.max_temp != 0)  # Any rows with max_temp not equal to zero?


# Resources
# 
# - [Learn Pandas](https://bitbucket.org/hrojas/learn-pandas)
# - [Compute](http://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/Cookbook%20-%20Compute.ipynb)
# - [Merge](http://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/Cookbook%20-%20Merge.ipynb)
# - [Select](http://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/Cookbook%20-%20Select.ipynb)
# - [Sort](http://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/Cookbook%20-%20Sort.ipynb)
# 
# 
# - [Intro to Pandas](https://bitbucket.org/hrojas/learn-pandas)
# - [Timeseries](http://nbviewer.ipython.org/github/changhiskhan/talks/blob/master/pydata2012/pandas_timeseries.ipynb)
# - [Statistics in Python](http://www.randalolson.com/2012/08/06/statistical-analysis-made-easy-in-python/)
# 
# http://datacommunitydc.org/blog/2013/07/python-for-data-analysis-the-landscape-of-tutorials/

# # Exploratory Data Analysis

# In[ ]:




# ## Plotting

# Some methods, like `plot()` and `hist()` produce plots using [matplotlib](http://matplotlib.org/).
# 
# To make plots using Matplotlib, you must first enable IPython's matplotlib mode. To do this, run the `%matplotlib inline` magic command to enable plotting in the current Notebook. \[If that doesn't work (because you have an older version of IPython), try `%pylab inline`. You may also have to restart the IPython kernel.\]
# 
# We'll go over plotting in more detail later.

# In[56]:

get_ipython().magic('matplotlib inline')
data.mean_temp.hist()


# In[57]:

ax = data.mean_temp.hist()   # get plot axes object
ax.set_xlabel('Daily Mean Temperature (F)')
ax.set_ylabel('# of Occurances')
ax.set_title('Mean Temperature Histogram')

fig = ax.get_figure()        # get plot figure object
fig.set_size_inches(8,6)     # set plot size
fig.savefig('MeanTempHistogram.png', dpi=100)


# We've already seen how the `hist()` function makes generating histograms a snap. Let's look at the `plot()` function now.

# In[58]:

data.max_temp.plot()


# That one line of code did a **lot** for us. First, it created a nice looking line plot using the maximum temperature column from our DataFrame. Second, because we used `datetime` objects in our index, pandas labeled the x-axis appropriately.
# 
# Pandas is smart too. If we're only looking at a couple of days, the x-axis looks different:

# In[59]:

data.max_temp.tail().plot()


# The `plot()` function returns a matplotlib `AxesSubPlot` object. You can pass this object into subsequent calls to `plot()` in order to compose plots.
# 
# Although `plot()` takes a variety of parameters to customize your plot, users familiar with matplotlib will feel right at home with the `AxesSubPlot` object.

# Prefer a bar plot? Pandas has got your covered.

# In[60]:

data.max_temp.tail().plot(kind="bar", rot=10)


# In[61]:

ax = data.max_temp.plot(title="Min and Max Temperatures")
data.min_temp.plot(style="red", ax=ax)
ax.set_ylabel("Temperature (F)")


# In[ ]:



