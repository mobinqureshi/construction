#!/usr/bin/env python
# coding: utf-8

# # 1. BUSINESS UNDERSTANDING

# “Like it or not, every construction company—and solutions provider—is now also in the data business. How well we help our customers transform that data into intelligence that drives better decisions to deliver projects more efficiently and more sustainably, with higher quality, lower costs and fewer risks is what defines the next frontier of construction management. Data is the key to improving the bottom line as well as protecting it. Our ability to break down data silos and transform raw data into action and intelligence is the crux to solving most challenges that rear their head in our industry. Solve the data problem and everything else falls into place.”—Jon Fingland, General Manager, Collaboration Solutions, Trimble
# 
# You have been tasked with analysing Ireland's Construction data and comparing the Irish Construction sector with other countries worldwide. This analysis should also include forecasting, sentiment analysis and evidence-based recommendations for the sector as well as a complete rationale of the entire process used to discover your findings. Your Research could include export, import, trade imbalance, house production, material stock, labour/skill pool, etc. (or any other relevant topic EXCEPT Climate change) with Ireland as your base line.
#  > Use relevant data to understand the market to improve construction company services and the business itself.
# 

# # 2. Data
# All the data used in this project was found in this site;("https://data.gov.ie/dataset?tags=construction") and ("https://www.kaggle.com/datasets/chicago/chicago-affordable-rental-housing-developments")

# ## Preparing the tools used in this project

# In[2]:


# Regular EDA and plotting libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import glob
import os
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA

# We want our plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# ## Import relavant dataset 

# ### 1. Social Housing Construction Status Report 2020 
# #### Social Housing Construction Status Report Q2 2020
# The Construction Status Q2 2020 report shows that 8,529 homes were under construction at end Q2 2020; and some 3,160 homes had been approved and were about to go on site. The full programme listed in the report now includes 2,154 schemes (or phases), delivering 29,810 homes – a substantial increase on the 22,139 homes which were in the programme at the end of Q2 2019
# Please note due to the Covid 19 Pandemic, Q1 2020 statistical returns were deferred to be collected in conjunction with Q2 2020. The Q1 Construction Status report is incorporated into the Q2 publication
# 
# #### Social Housing Construction Status Report Q3 2020
# The Construction Status Q3 2020 report shows that 9,562 homes were under construction at end Q3 2020; and some 3,133 homes had been approved and were about to go on site. The full programme listed in the report now includes 2,283 schemes (or phases), delivering 31,862 homes – a substantial increase on the 22,721 homes which were in the programme at the end of Q3 201
# 
# #### Social Housing Construction Status Report Q4 2020
# The Minister recently published the Construction Status Report (CSR) for Quarter 4 2020. The CSR provides scheme level detail on new build social housing activity in each local authority area.
# Commenting on the report Minister O’Brien said, “The report shows a strong pipeline for new social homes with 8,555 social homes on site and over 9,000 homes at various stages of the approval process. The key priority for my Department is increasing the supply of social housing, I intend to publish our new housing plan ‘Housing for All’ later this summer. It will build on our commitments in the Programme for Government and provide a roadmap, with a whole of Government approach, to outline how we get to a housing system that gives us the sustainable supply we need, at a price that people can afford, with appropriate housing options for the most vulnerable in our society.”
# 

# In[3]:


# Set the directory containing the datasets
data_dir = 'C:/Users/OMBATI/Desktop/python codes/project money/social_housing_construction_status_report_2020'

# Get a list of all the CSV files in the directory
csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

# Initialize an empty list to store the dataframes
dfs = []

# Loop through each CSV file and read it into a dataframe
for file in csv_files:
    filepath = os.path.join(data_dir, file)
    df = pd.read_csv(filepath)
    dfs.append(df)

# Concatenate all the dataframes into a single dataframe
social_housing = pd.concat(dfs, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv('social_housing5.csv', index=False)

# Print the firts few rows of the combined dataframe
social_housing.head()


# The  three datasets were combned into one dataset called *social_housing*

# ## 2. National House Construction Cost Index

# (https://data.gov.ie/dataset/national-house-construction-cost-index?package_type=dataset)
# 
# The index relates to costs ruling on the first day of each month.
# 
# NATIONAL HOUSE CONSTRUCTION COST INDEX; Up until October 2006 it was known as the National House Building Index Oct 2000 data; The index since October, 2000, includes the first phase of an agreement following a review of rates of pay and grading structures for the Construction Industry and the first phase increase under the PPF.
# 
# April, May and June 2001; Figures revised in July 2001due to 2% PPF Revised Terms. March 2002; The drop in the March 2002 figure is due to a decrease in the rate of PRSI from 12% to 10¾% with effect from 1 March 2002.
# 
# The index from April 2002 excludes the one-off lump sum payment equal to 1% of basic pay on 1 April 2002 under the PPF.
# April, May, June 2003; Figures revised in August'03 due to the backdated increase of 3% from 1April 2003 under the National Partnership Agreement 'Sustaining Progress'.
# 
# The increases in April and October 2006 index are due to Social Partnership Agreement "Towards 2016".
# March 2011; The drop in the March 2011 figure is due to a 7.5% decrease in labour costs.
# Methodology in producing the Index
# 
# Prior to October 2006:
# The index relates solely to labour and material costs which should normally not exceed 65% of the total price of a house. It does not include items such as overheads, profit, interest charges, land development etc.
# The House Building Cost Index monitors labour costs in the construction industry and the cost of building materials. It does not include items such as overheads, profit, interest charges or land development. The labour costs include insurance cover and the building material costs include V.A.T. Coverage:
# The type of construction covered is a typical 3 bed-roomed, 2 level local authority house and the index is applied on a national basis.
# 
# Data Collection:
# The labour costs are based on agreed labour rates, allowances etc. The building material prices are collected at the beginning of each month from the same suppliers for the same representative basket.
# 
# Calculation:
# Labour and material costs for the construction of a typical 3 bed-roomed house are weighted together to produce the index.
# 
# Post October 2006:
# The name change from the House Building Cost Index to the House Construction Cost Index was introduced in October 2006 when the method of assessing the materials sub-index was changed from pricing a basket of materials (representative of a typical 2 storey 3 bedroomed local authority house) to the CSO Table 3 Wholesale Price Index. The new Index does maintains continuity with the old HBCI.
# 
# The most current data is published on these sheets. Previously published data may be subject to revision. Any change from the originally published data will be highlighted by a comment on the cell in question. These comments will be maintained for at least a year after the date of the value change. Oct 2008 data; Decrease due to a fall in the Oct Wholesale Price Index.

# In[316]:


# Set the path to the directory containing the dataset
data_dir = 'C:/Users/OMBATI/Desktop/python codes/project money'

# Set the filename of the dataset
filename = 'national_house_construction_cost_index_0.csv'

# Combine the directory and filename to get the full filepath
filepath = os.path.join(data_dir, filename)

# Read the dataset into a pandas DataFrame
national_house = pd.read_csv(filepath)

# Print the first few rows of the DataFrame to check that it loaded correctly
national_house.T.head()


# ## 3. BEQ04 - Indices of Total Production in Building and Construction Sector (Base 2015=100)
# Indices of Total Production in Building and Construction Sector (Base 2015=100).

# In[317]:


annual_employment


# ## 4.Chicago Affordable Rental Housing Developments

# ## About Dataset
# 
# Content 
# 
# Thousands of affordable housing developments, backed by City of Chicago programs, are being built, including the rental housing developments listed below. When new projects finish construction or when the compliance time for existing projects expires, usually after 30 years, the list is frequently updated.
# 
# The public is kindly offered access to the list. It excludes the hundreds of thousands of naturally existing affordable housing units spread around Chicago that are not supported by the City, as well as any affordable housing units that may be available for rent that are assisted by the City. Contact each property separately for details on rents, minimum income requirements, and availability for the projects listed. Call (877) 428-8844 or go to www.ILHousingSearch.org for more on additional reasonably priced rental homes in Chicago and Illinois. Context The City of Chicago is the host of this dataset.
# 
# The city has an open data platform that can be seen here, and they update their data in accordance with the volume of data received. Utilize Kaggle and all the data sources offered on the City of Chicago organization website to explore the city! This dataset is updated on a monthly basis..

# In[14]:


# Set the path to the directory containing the dataset
data_dir = 'C:/Users/OMBATI/Desktop/python codes/project money'

# Set the filename of the dataset
filename = 'affordable-rental-housing-developments.csv'

# Combine the directory and filename to get the full filepath
filepath = os.path.join(data_dir, filename)

# Read the dataset into a pandas DataFrame
affordable = pd.read_csv(filepath)

# Print the first few rows of the DataFrame to check that it loaded correctly
affordable.head()


# # 1. Social Housing Construction Status Report 2020

# ## EXPLORATORY DATA ANALYSIS

# ## First rows

# In[318]:


social_housing


# The combined dataset has 6820rows and 13 columns.

# In[30]:


social_housing.tail()


# In[320]:


social_housing.T.head()


# # Data types

# In[321]:


social_housing.info()


# There are 3 numeric data type and 10 object data type

# # Names of columns

# In[322]:


print(social_housing.columns)


# ## Missing data

# In[323]:


social_housing.isna().sum()


# ## Some data cleaning before proceeding

# In[324]:


# Loop through all object columns and convert to categorical data type
for col in social_housing.select_dtypes(include='object').columns:
    social_housing[col] = social_housing[col].astype('category')

# Print the data types of all columns
print(social_housing.dtypes)


# In[325]:


# Value count of Funding Programme 
social_housing.iloc[:, 1].value_counts()


# In[326]:


# Plot the value counts with a bar graph
social_housing.iloc[:, 1].value_counts().plot(kind="bar");


# In[327]:


# Value count of LA
social_housing.iloc[:, 2].value_counts()


# In[328]:


# visualization of value count of LA
social_housing.iloc[:, 2].value_counts().plot(kind='bar');


# In[329]:


# value count of Scheme/Project Name
social_housing.iloc[:, 3].value_counts()


# In[330]:


# visualization of value count of Scheme/Project Name
social_housing.iloc[:, 3].value_counts().plot(kind="bar");


# In[331]:


# Value count of Stage 1 Capital Appraisal
social_housing.iloc[:, 6].value_counts()


# In[332]:


# Value count of Stage 1 Capital Appraisal visualization
social_housing.iloc[:, 6].value_counts().plot(kind="bar");


# In[333]:


# Value count of Stage 2 Pre Planning
social_housing.iloc[:, 7].value_counts()


# In[334]:


# Value count of Stage 2 Pre Planning visualization
social_housing.iloc[:, 7].value_counts().plot(kind="bar");


# In[335]:


# Value count of Stage 3 Pre Tender design
social_housing.iloc[:, 8].value_counts()


# In[336]:


# Value count of Stage 3 Pre Tender design visualization
social_housing.iloc[:, 8].value_counts().plot(kind="bar");


# In[337]:


# Value count of Stage 4 Tender Report or Final Turnkey/CALF approval
social_housing.iloc[:, 9].value_counts()


# In[338]:


# Value count of Stage 4 Tender Report or Final Turnkey/CALF approval visualition
social_housing.iloc[:, 9].value_counts().plot(kind="bar");


# In[339]:


# Value count of On Site
social_housing.iloc[:, 10].value_counts()


# In[340]:


# Value count of On Site visualiation
social_housing.iloc[:, 10].value_counts().plot(kind="bar");


# In[341]:


# Value count of  Completed
social_housing.iloc[:, 11].value_counts()


# In[342]:


# Value count of  Completed visuliation 
social_housing.iloc[:, 11].value_counts().plot(kind="bar");


# In[343]:


# No_ histrogram
social_housing.iloc[:, 12].value_counts().hist();


# In[344]:


social_housing.head()


# ### Plot a histogram of the number of units

# In[345]:


plt.hist(social_housing['No_ of Units'], bins=20)
plt.show()


# ### Group the data by LA and calculate the mean number of units

# In[346]:


la_means = social_housing.groupby('LA')['No_ of Units'].mean()
print(la_means)


# ## National House Construction Cost Index

# In[347]:


national_house


# ### Making the first column as the heading of the dataset

# In[348]:


# extract the third row as column names
new_header = national_house.iloc[1]

# set the third row as the column names and drop the first two rows
national_house = national_house[2:]
national_house.columns = new_header

# reset the index
national_house = national_house.reset_index(drop=True)

national_house


# In[349]:


national_house.T


# In[350]:


national_house.shape


# In[351]:


national_house.info()


# In[352]:


national_house.describe()


# In[353]:


# convert all object columns to numeric
df = national_house.apply(pd.to_numeric, errors='coerce')

# print the data types of each column after conversion
print(df.dtypes)


# In[354]:


e = df.describe()
e.T


# In[355]:


national_house


# In[ ]:





# In[356]:


national_house.info()


# In[357]:


## checking missing values
national_house.isna().sum()


# In[358]:


# Replace all NAs with 0
national_house = national_house.fillna(0)
national_house.isna().sum()


# ## Box Plot of Years

# In[359]:


# Convert object columns to float
for col in national_house.columns:
    if col != 'Month':
        national_house[col] = pd.to_numeric(national_house[col], errors='coerce')


(national_house.dtypes)


# In[360]:


national_house


# In[361]:


plt.figure(figsize=(12, 8)) # set figure size
sns.boxplot(data=national_house.iloc[:-1, 1:])
plt.title('Years')
plt.show()


# ### Heatmap of Correlations between Yealy Prices

# In[362]:


plt.figure(figsize=(12, 8)) # set figure size
corr_matrix = national_house.iloc[:-1, 1:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of yearly Prices')
plt.show()


# In[363]:


corr_matrix


# In[364]:


national = national_house.T

# Set the second row as the column headers
national.columns = national.iloc[0]
national = national.drop(national.index[0])

# Convert all columns to numeric
national = national.apply(pd.to_numeric, errors='coerce')
national 


# In[365]:


national.info()


# In[366]:


plt.figure(figsize=(12, 8)) # set figure size
sns.boxplot(data=national.iloc[:, :])
plt.title('Monthly Prices for Each Year')
plt.show()


# In[367]:


national['Yearly average'].plot(kind="line")


# ## Hypothesis Testing
# ### One-sample t-test to determine if the average price in January 1994 was significantly different from the population mean

# In[368]:


national


# In[369]:


jan_1994_price = national.iloc[0, 0]
pop_mean = national.iloc[:, :-2].values.mean()


# In[370]:


t_stat, p_val = ttest_1samp(national.iloc[:, 1], pop_mean)
alpha = 0.05


# In[371]:


if p_val < alpha:
    print(f"The p-value ({p_val}) is less than the significance level ({alpha}), so we reject the null hypothesis.")
else:
    print(f"The p-value ({p_val}) is greater than the significance level ({alpha}), so we fail to reject the null hypothesis.")


# The one-sample t-test was conducted to determine if the average price in January 1994 was significantly different from the population mean. The null hypothesis states that the sample mean is not significantly different from the population mean, while the alternative hypothesis states that the sample mean is significantly different from the population mean.
# 
# The result of the hypothesis test showed that the p-value (0.45618455962118665) is greater than the significance level (0.05), so we fail to reject the null hypothesis. This means that we do not have enough evidence to conclude that the average price in January 1994 is significantly different from the population mean.
# 
# In other words, the test did not find evidence to support the claim that the average price in January 1994 was significantly different from the population mean at the 5% level of significance. Therefore, we cannot conclude that the price in January 1994 was either significantly higher or significantly lower than the population mean.

# ### Two-sample t-test to determine if the average price in January 1994 was significantly different from the average price in January 2001

# In[372]:


from scipy.stats import ttest_ind

jan_1994_price = national.iloc[0, 0]
jan_2001_price = national.iloc[7, 0]


# In[373]:


t_stat, p_val = ttest_ind(jan_1994_price,  jan_2001_price)
alpha = 0.05


# In[374]:


if p_val < alpha:
    print(f"The p-value ({p_val}) is less than the significance level ({alpha}), so we reject the null hypothesis.")
else:
    print(f"The p-value ({p_val}) is greater than the significance level ({alpha}), so we fail to reject the null hypothesis.")


# The given report indicates that a two-sample t-test was conducted to determine if the average price in January 1994 was significantly different from the average price in January 2001. The null hypothesis was likely that there was no significant difference between the two average prices. The p-value obtained from the t-test was reported as "nan" which stands for "not a number".
# 
# A p-value greater than the significance level (0.05) was also reported, which indicates that there was not enough evidence to reject the null hypothesis. In other words, there was not enough evidence to suggest that the average price in January 1994 was significantly different from the average price in January 2001.
# 
# However, it is important to note that a p-value of "nan" is not a valid result and may indicate an issue with the data or the analysis. It is possible that there were missing or invalid values in the dataset, or that the data was not properly formatted for the analysis. Therefore, it is important to investigate the reason for the "nan" value and to ensure that the analysis is valid before making any conclusions based on the results.

# ### Chi-squared test to determine if there is a significant association between the month and the year of the prices

# In[375]:


from scipy.stats import chi2_contingency

month_year_ct = pd.crosstab(national_house.iloc[:, :-2].stack(), national_house.iloc[:, :-2].stack().index.get_level_values(1))
chi2, p_val, dof, expected = chi2_contingency(month_year_ct)

alpha = 0.05

if p_val < alpha:
    print(f"The p-value ({p_val}) is less than the significance level ({alpha}), so we reject the null hypothesis.")
else:
    print(f"The p-value ({p_val}) is greater than the significance level ({alpha}), so we fail to reject the null hypothesis.")


# The result of the chi-squared test shows that the p-value (2.7147638050442284e-33) is less than the significance level of 0.05. This means that there is sufficient evidence to reject the null hypothesis, which suggests that there is no significant association between the month and the year of the prices.
# 
# In other words, the test indicates that there is a significant association between the month and year of the prices. This means that the month and year variables are not independent and that they have some degree of association with each other.

# ### Predicting January values from 2017 upto 2027 values using ARIMA model

# In[376]:


train_data = national.iloc[:-1, 12] 


# In[377]:


model = ARIMA(train_data, order=(1, 1, 1)) 


# In[378]:


model_fit = model.fit() 


# In[379]:


# Predict values for 
start_index = len(train_data)
end_index = start_index + 11 
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')


# In[380]:


# Print predicted values from year 2017
print("Predicted values for year  2017 upto 2027 :")
(predictions)


# # BEQ04 - Indices of Total Production in Building and Construction Sector (Base 2015=100)

# In[381]:


annual_employment


# In[382]:


annual_employment.T


# In[383]:


annual_employment.info()


# ## Convert all object column into categorical

# In[384]:


# Get object column names
object_cols = list(annual_employment.select_dtypes(include=["object"]).columns)

# Convert object columns to categorical
annual_employment[object_cols] = annual_employment[object_cols].astype("category")

# Verify the data type conversion
print(annual_employment.dtypes)


# In[385]:


## Check missing data
annual_employment.isna().sum()


# It has no missing data

# In[386]:


## summary statistics of the data
annual_employment.describe()


# In[387]:


## Plot the distribution of the data
plt.hist(annual_employment["VALUE"])
plt.show()


# In[388]:


## Plot the time series of the data
plt.plot(annual_employment["Year"], annual_employment["VALUE"])
plt.show()


# ### Correlation matrix

# In[389]:


corr_matrix = annual_employment.corr()
print('Correlation matrix:')
print(corr_matrix)


# ### Scatter plot

# In[390]:


plt.scatter(annual_employment['Year'], annual_employment['VALUE'])
plt.title('Scatter Plot of Employment Index')
plt.xlabel('Year')
plt.ylabel('Index Value')
plt.show()


# ## Hypothesis test
# ### Test if the mean value of the data is equal to 100

# In[391]:


t_test = np.mean(annual_employment["VALUE"]) - 100
p_value = ttest_1samp(annual_employment["VALUE"], 100).pvalue
print("t-statistic:", t_test)
print("p-value:", p_value)


# ### Test if the mean value of the data is different from 100

# In[392]:


t_test = np.mean(annual_employment["VALUE"]) - 100
p_value = ttest_1samp(annual_employment["VALUE"], 100).pvalue
if p_value < 0.05:
    print("The mean value of the data is different from 100")
else:
    print("The mean value of the data is not different from 100")


# ### Test if the data is normally distributed

# In[393]:


from scipy.stats import shapiro
shapiro_test = shapiro(annual_employment["VALUE"])
p_value = shapiro_test.pvalue
if p_value < 0.05:
    print("The data is not normally distributed")
else:
    print("The data is normally distributed")


# ## Random regression

# In[394]:


# Regression model
x = sm.add_constant(annual_employment['Year'])
y = annual_employment['VALUE']
model = sm.OLS(y, x).fit()
print('Regression model summary:')
print(model.summary())


# In[397]:


# Residual plot
residuals = model.resid
plt.scatter(annual_employment['Year'], residuals)
plt.title('Residual Plot of Regression Model')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.show()


# In[398]:


#Random regression
np.random.seed(123)
random_year = np.random.randint(1975, 2006, size=len(data))
random_x = sm.add_constant(random_year)
random_y = annual_employment['VALUE']
random_model = sm.OLS(random_y, random_x).fit()
print('Random regression model summary:')
print(random_model.summary())


# In[399]:


#Random regression
np.random.seed(123)
random_year = np.random.randint(1975, 2006, size=len(data))
random_x = sm.add_constant(random_year)
random_y = annual_employment['VALUE']
random_model = sm.OLS(random_y, random_x).fit()
print('Random regression model summary:')
print(random_model.summary())


# ## Chicago Affordable Rental Housing Developments

# In[15]:


affordable


# In[17]:


affordable.info()


# In[19]:


float_cols = ['Community Area Number', 'Zip Code', 'Units', 'X Coordinate', 'Y Coordinate', 
              'Latitude', 'Longitude', 'Historical Wards 2003-2015', 'Wards', 'Community Areas', 
              'Zip Codes', 'Census Tracts']

affordable[float_cols] = affordable[float_cols].astype(float)


# In[21]:


cat_cols = ['Phone Number','Community Area Name', 'Property Type', 'Property Name', 'Address', 'Management Company']

affordable[cat_cols] = affordable[cat_cols].astype('category')


# In[22]:


affordable.info()


# In[23]:


affordable


# In[25]:


# check missing values
affordable.isna().sum()


# In[26]:


# Drop rows with missing data in 'Latitude' and 'Longitude'
affordable.dropna(subset=['Latitude', 'Longitude'], inplace=True)


# In[27]:


affordable


# In[32]:


social_housing_df = social_housing
affordable_df = affordable


# In[33]:


# Check the data types of the columns.
social_housing_df.dtypes
affordable_df.dtypes


# In[34]:


# Create a new column in each DataFrame to indicate the country of origin.
social_housing_df['Country'] = 'Ireland'
affordable_df['Country'] = 'Chicago'


# In[38]:


social_housing_df.head()


# In[39]:


affordable_df.head()


# In[40]:


# Run a t-test to compare the mean number of units per project between Ireland and Chicago.
t_test, p_value = stats.ttest_ind(social_housing_df['No_ of Units'].loc[social_housing_df['Country'] == 'Ireland'], affordable_df['Units'].loc[affordable_df['Country'] == 'Chicago'])
print('t-test:', t_test)
print('p-value:', p_value)


# The t-test result of -40.92851928381021 means that the difference between two groups or samples being compared is very large, and it is unlikely to have occurred by chance.
# 
# The p-value of 0.0 indicates that the probability of obtaining a t-statistic as extreme as -40.92851928381021, assuming that the null hypothesis (the two groups being compared are not different) is true, is extremely low. Typically, if the p-value is less than 0.05 (5%), it is considered statistically significant, meaning that the null hypothesis can be rejected and that there is evidence of a significant difference between the two groups. In this case, since the p-value is 0.0, we can conclude that there is a highly significant difference between the two groups being compared.

# In[52]:


from scipy.stats import chi2_contingency

# Create a contingency table of the two columns being compared
cont_table = pd.crosstab(df_ireland["Funding Programme"], df_chicago["Property Type"])

# Perform a chi-squared test on the contingency table
chi2_stat, p_value, dof, expected = chi2_contingency(cont_table)

# Print the test results
print("Chi-square statistic:", chi2_stat)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)


# In[56]:


from scipy.stats import kruskal

# Create a list of the median number of units completed in each community area in Ireland
ireland_medians = df_ireland.groupby("Funding Programme")["No_ of Units"].median().tolist()

# Create a list of the median number of units completed in each community area in Chicago
chicago_medians = df_chicago.groupby("Community Area Name")["Units"].median().tolist()

# Perform a Kruskal-Wallis H test on the two lists of medians
h_stat, p_value = kruskal(ireland_medians, chicago_medians)

# Print the test results
print("Kruskal-Wallis H statistic:", h_stat)
print("p-value:", p_value)


# In[44]:


# Run a Mann-Whitney U test to compare the median number of units completed in Ireland and Chicago
u_test = stats.mannwhitneyu(df_ireland["No_ of Units"], df_chicago["Units"])
print(u_test)

