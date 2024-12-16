## Introduction:

+ I am working as a data scientist for a credit card company, where the Data Science Coordinator has proposed using the K-means algorithm to segment customer data

## Goals

+ The goal is to segment customers into distinct groups, enabling the company to apply tailored business strategies for each customer type.
+ The groups will vary based on factors such as credit limits, purchasing patterns, and trends. Higher-income users will be given higher credit limits, while those who use their credit cards less will be incentivized to increase usage.
+ The company expects to receive a report detailing the segmentation for each client, along with an explanation of the characteristics of each group and the key differentiators that set them apart.

## Dataset contains:

+ `customer_id`: unique identifier for each customer
+ `age`: customer age in years
+ `gender`: customer gender (M or F)
+ `dependent_count`: number of dependents of each customer
+ `education_level`: level of education ("High School", "Graduate", etc.)
+ `marital_status`: marital status ("Single", "Married", etc.)
+ `estimated_income`: the estimated income for the customer projected by the data science team
+ `months_on_book`: time as a customer in months
+ `total_relationship_count`: number of times the customer contacted the company
+ `months_inactive_12_mon`: number of months the customer did not use the credit card in the last 12 months
+ `credit_limit`: customer's credit limit
+ `total_trans_amount`: the overall amount of money spent on the card by the customer
+ `total_trans_count`: the overall number of times the customer used the card
+ `avg_utilization_ratio`: daily average utilization ratio



```python
# Importing pandas library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reading the CSV file into a pandas dataframe
data = pd.read_csv('customer_segmentation.csv')
print(data.head(2))
```

       customer_id  age gender  dependent_count education_level marital_status  \
    0    768805383   45      M                3     High School        Married   
    1    818770008   49      F                5        Graduate         Single   
    
       estimated_income  months_on_book  total_relationship_count  \
    0             69000              39                         5   
    1             24000              44                         6   
    
       months_inactive_12_mon  credit_limit  total_trans_amount  \
    0                       1       12691.0                1144   
    1                       1        8256.0                1291   
    
       total_trans_count  avg_utilization_ratio  
    0                 42                  0.061  
    1                 33                  0.105  



```python
# shape of the data:
print(f'Rows and Columns:{data.shape}')
```

    Rows and Columns:(10127, 14)



```python
# Data Types of the Dataset
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10127 entries, 0 to 10126
    Data columns (total 14 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   customer_id               10127 non-null  int64  
     1   age                       10127 non-null  int64  
     2   gender                    10127 non-null  object 
     3   dependent_count           10127 non-null  int64  
     4   education_level           10127 non-null  object 
     5   marital_status            10127 non-null  object 
     6   estimated_income          10127 non-null  int64  
     7   months_on_book            10127 non-null  int64  
     8   total_relationship_count  10127 non-null  int64  
     9   months_inactive_12_mon    10127 non-null  int64  
     10  credit_limit              10127 non-null  float64
     11  total_trans_amount        10127 non-null  int64  
     12  total_trans_count         10127 non-null  int64  
     13  avg_utilization_ratio     10127 non-null  float64
    dtypes: float64(2), int64(9), object(3)
    memory usage: 1.1+ MB
    None



```python
# Columns with categorical data
print(data.select_dtypes(include=['object']).columns)
```

    Index(['gender', 'education_level', 'marital_status'], dtype='object')



```python
# Checking for null values
print(f'Presence of null values:\n{data.isna().sum()}')
```

    Presence of null values:
    customer_id                 0
    age                         0
    gender                      0
    dependent_count             0
    education_level             0
    marital_status              0
    estimated_income            0
    months_on_book              0
    total_relationship_count    0
    months_inactive_12_mon      0
    credit_limit                0
    total_trans_amount          0
    total_trans_count           0
    avg_utilization_ratio       0
    dtype: int64



```python
# Categorical columns count
gender_count = pd.DataFrame(data['gender'].value_counts().reset_index())
education_count = pd.DataFrame(data['education_level'].value_counts().reset_index())
marital_count = pd.DataFrame(data['marital_status'].value_counts().reset_index())
```


```python
# Plotting charts for categorical columns
fig,(ax) = plt.subplots(3,1,figsize=(8,8))
gender_count.plot(kind='bar',ax=ax[0],color = 'grey')  # int data type
ax[0].tick_params(axis='x', labelrotation=0)
ax[0].set_xlabel('Gender')
ax[0].set_ylabel('Gender_count')
ax[0].set_title('Gender_count')
for index, value in enumerate(gender_count['count']):
    ax[0].text(index, value + 50, str(value), ha='center')
    
education_count.plot(kind='bar',ax=ax[1],color ='orange') # Object data type
ax[1].set_xlabel('Education')
ax[1].set_ylabel('Education_count')
ax[1].tick_params(axis='x', labelrotation=0)
ax[1].set_title('Education_count')
for ind, val in enumerate(education_count['count']):
    ax[1].text(ind, val + 50, str(val), ha='center')

marital_count.plot(kind='bar',ax=ax[2])  # Object data type
ax[2].set_xlabel('Marital_Status')
ax[2].set_ylabel('Marital_Status_count')
ax[2].tick_params(axis='x', labelrotation=0)
ax[2].set_title('Marital_status_count')
for ind, val in enumerate(marital_count['count']):
    ax[2].text(ind, val + 50, str(val), ha='center')

plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    


## Distribution of Age
+ The age distribution is approximately symmetric, with values evenly distributed around the mean and a skewness of -0.03360.
+ The age range spans from 26 to 73, with an average age of 46. The maximum age is 67 for females and 73 for males, while the minimum age is 26 for both genders.


```python
print(data.groupby('gender').agg(mean_age=('age','mean'),max_age =('age','max'),min_age =('age','min')).round())
```

            mean_age  max_age  min_age
    gender                            
    F           46.0       67       26
    M           46.0       73       26



```python
# Distribution of 'age' field is nearly symmetric and skewness closer to zero
# skewness rate
skewness = data['age'].skew()
print(f'Skewness: {skewness}')

data['age'].hist(figsize=(10,6),color='lightgrey', edgecolor='black')
plt.title('Age Distribution')  
plt.xlabel('Age')         # X-axis label
plt.ylabel('Frequency')           # Y-axis label
plt.grid(axis='y', alpha=0.75)         
plt.show()
```

    Skewness: -0.033605016317173456



    
![png](output_12_1.png)
    



```python
## Disribution of Numerical features by gender
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
data.boxplot(column= 'age',by ='gender',grid=True, figsize=(8,6))
plt.suptitle('')
plt.title('Box Plot of Age by Gender category')
plt.xlabel('Gender')                     # X-axis label
plt.ylabel('Age')                       # Y-axis label
plt.show()
```


    
![png](output_13_0.png)
    


## Distribution of Dependents
+ The dependent count is approximately symmetric, meaning the values are evenly distributed around the mean, with a skewness of -0.02082.
+ The average number of dependents is 2 for both genders, with a maximum count of 5.


```python
print(data.groupby('gender').agg(mean_dep_count=('dependent_count','mean'), max_dep_count =('dependent_count','max'),min_dep_count =('dependent_count','min')).round())
```

            mean_dep_count  max_dep_count  min_dep_count
    gender                                              
    F                  2.0              5              0
    M                  2.0              5              0



```python
# Distribution of 'age' field is nearly symmetric and skewness closer to zero
# skewness rate
skewness = data['dependent_count'].skew()
print(f'Skewness: {skewness}')

# Histogram
data['dependent_count'].hist(figsize=(10,6),color='orange', edgecolor='black')
plt.title('Dependent Distribution')  
plt.xlabel('Dependents')         # X-axis label
plt.ylabel('Frequency')          # Y-axis label
plt.grid(axis='y', alpha=0.75)         
plt.show()
```

    Skewness: -0.020825535616339912



    
![png](output_16_1.png)
    


## Distribution of Estimated income
+ The income ranges from a minimum of 20k to a maximum of 200k
+ Income is not evenly distributed between genders. The mean income for females is 39K, while for males it is 87K
+ Out of 10,127 observations, 315 women earn more than 75,000, while 2616 men earn more than $75,000
+ The data for estimated income is heavily right-skewed, with a skewness value of 1.3615


```python
print(f'Min, Max and Mean value of income:\n{data['estimated_income'].describe().round()[['min','max','mean']]}')
```

    Min, Max and Mean value of income:
    min      20000.0
    max     200000.0
    mean     62078.0
    Name: estimated_income, dtype: float64


### Income distribution based on Gender


```python
print(data.groupby('gender').agg(mean_income =('estimated_income','mean'),max_inc =('estimated_income','max'),min_inc =('estimated_income','min')).round())
```

            mean_income  max_inc  min_inc
    gender                               
    F           39725.0   200000    20000
    M           87192.0   200000    20000



```python
print(f'''Total Women who earn above 75000: {data[(data['gender']=='F')&(data['estimated_income']>=75000)]['customer_id'].count()}''')
print(f'''Total Men who earn above 75000: {data[(data['gender']=='M')&(data['estimated_income']>=75000)]['customer_id'].count()}''')
```

    Total Women who earn above 75000: 315
    Total Men who earn above 75000: 2616



```python
# Charts to plot the estimated income
# Importing pandas library
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
data.boxplot(column= 'estimated_income',by ='gender',grid=True, figsize=(8,6))
plt.suptitle('')
plt.title('Box Plot of estimated_income by Gender category')
plt.xlabel('Gender')                     # X-axis label
plt.ylabel('Estimated_income')           # Y-axis label
plt.show()
```


    
![png](output_22_0.png)
    



```python
# skewness rate
skewness = data['estimated_income'].skew()
print(f'Skewness: {skewness}')

# Histogram
data.hist(column = 'estimated_income',figsize=(10,6),color='lightblue', edgecolor='black')
plt.title('Distribution of Estimated income')
plt.show()
```

    Skewness: 1.3615188240734424



    
![png](output_23_1.png)
    


## Distribution of months_on_book


```python
# skewness rate
skewness = data['months_on_book'].skew()
print(f'Skewness: {skewness}')

import seaborn as sns
sns.kdeplot(data['months_on_book'],shade=True)
plt.show()
```

    Skewness: -0.10656535989402989


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/411287705.py:6: FutureWarning: 
    
    `shade` is now deprecated in favor of `fill`; setting `fill=True`.
    This will become an error in seaborn v0.14.0; please update your code.
    
      sns.kdeplot(data['months_on_book'],shade=True)



    
![png](output_25_2.png)
    


## Distribution of months_inactive_12_mon by customers
+ The distribution is skewed to the right with a skewness value of 0.6330
+ There may be an opportunity to engage customers who have remained inactive for more than 4 months, particularly those with higher credit limits


```python
months_inactive = data['months_inactive_12_mon'].value_counts().reset_index().sort_values(by='months_inactive_12_mon')
print(months_inactive)
```

       months_inactive_12_mon  count
    6                       0     29
    2                       1   2233
    1                       2   3282
    0                       3   3846
    3                       4    435
    4                       5    178
    5                       6    124



```python
# skewness rate
skewness = data['months_inactive_12_mon'].skew()
print(f'Skewness: {skewness}')

# Histogram
months_inactive.plot(kind='bar',x = 'months_inactive_12_mon',y='count',figsize=(10,6),color='lightblue', edgecolor='black')
plt.title('Distribution of Months inactive')
for i,j in enumerate(months_inactive['count']):
    plt.text(i, j + 50, str(j), ha='center')
plt.show()
```

    Skewness: 0.6330611289713137



    
![png](output_28_1.png)
    


## Distribution of Customer Relationship count, Dependent count and Inactive_months


```python
# Value counting features
total_relationship_count = data['total_relationship_count'].value_counts().reset_index()
Dependent_count = data['dependent_count'].value_counts().reset_index()
Inactive_months = data['months_inactive_12_mon'].value_counts().reset_index()
```


```python
# Plotting charts 
fig,(ax) = plt.subplots(3,1,figsize=(10,10))
total_relationship_count.plot(kind='bar',ax=ax[0],color = 'grey')  # int data type
ax[0].tick_params(axis='x', labelrotation=0)
ax[0].set_xlabel('No of times customer contacted the company')
ax[0].set_ylabel('contact_count')
ax[0].set_title('Times customer contacted the company')
for index, value in enumerate(total_relationship_count['count']):
    ax[0].text(index, value + 50, str(value), ha='center')
    
Dependent_count.plot(kind='bar',ax=ax[1],color ='orange') # Object data type
ax[1].set_xlabel('Total dependents')
ax[1].set_ylabel('count')
ax[1].tick_params(axis='x', labelrotation=0)
ax[1].set_title('Customers with dependents count')
for ind, val in enumerate(Dependent_count['count']):
    ax[1].text(ind, val + 50, str(val), ha='center')

Inactive_months.plot(kind='bar',ax=ax[2])  # Object data type
ax[2].set_xlabel('Inactive months')
ax[2].set_ylabel('Inactive_user_count')
ax[2].tick_params(axis='x', labelrotation=0)
ax[2].set_title('Inactive_user_count')
for ind, val in enumerate(Inactive_months['count']):
    ax[2].text(ind, val + 50, str(val), ha='center')

plt.tight_layout()
plt.show()
```


    
![png](output_31_0.png)
    


## Credit Limit
+ The credit limit ranges from a minimum of 1,000 dollars to a maximum of 34,000
+ The credit limit varies based on the customer's income
+ To better understand the credit limits across different income levels, I divided the data into income bins
+ The average credit limit varies by income level as follows:
    + 0–40k: 4,360.0
    + 40k–80k: 7,995.0
    + 80k–120k: 14,994.0
    + 120k–160k: 18,588.0
    + 160k–200k: 19,159.0

+ As income increases, the average credit limit also increases.


```python
print(f'Minimum and Maximum credit limit:\n{data['credit_limit'].describe()[['min','max']]}')
```

    Minimum and Maximum credit limit:
    min     1438.3
    max    34516.0
    Name: credit_limit, dtype: float64



```python
#Creating income bins for analysis:
bins = [0, 40000, 80000, 120000, 160000, 200000]
bin_labels = ['0-40k', '40k-80k', '80k-120k', '120k-160k', '160k-200k']

#Create a new column 'income_bin' that categorizes income
data['income_bin'] = pd.cut(data['estimated_income'], bins=bins, labels=bin_labels, right=False)

#Grouping data using the income bins
income_group = data.groupby('income_bin').agg(total_count = ('income_bin','count')).reset_index()
```

    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/2346477604.py:9: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      income_group = data.groupby('income_bin').agg(total_count = ('income_bin','count')).reset_index()


### Distribution of customer by Income bins


```python
# income Bins and the customer count
income_group.plot(kind='bar',color = 'grey')  # int data type
plt.tick_params(labelrotation=0)
plt.xlabel('Income Bin')
plt.ylabel('Count')
plt.xticks(ticks=[0,1,2,3,4], labels=['0-40k', '40k-80k', '80k-120k', '120k-160k', '160k-200k'])
for index, value in enumerate(income_group['total_count']):
    plt.text(index, value + 50, str(value), ha='center')
    
plt.show()
```


    
![png](output_36_0.png)
    


### Mean credit limit based on Income bins


```python
print(data.groupby('income_bin').agg(Mean_credit_limit=('credit_limit','mean'),Median_credit_limit=('credit_limit','median'),Max_credit_limit=('credit_limit','max'),count_credit_limit=('credit_limit','count')).round())
```

                Mean_credit_limit  Median_credit_limit  Max_credit_limit  \
    income_bin                                                             
    0-40k                  4360.0               2882.0           34516.0   
    40k-80k                7995.0               5104.0           34516.0   
    80k-120k              14994.0              11808.0           34516.0   
    120k-160k             18588.0              16750.0           34516.0   
    160k-200k             19159.0              17572.0           34516.0   
    
                count_credit_limit  
    income_bin                      
    0-40k                     3981  
    40k-80k                   3601  
    80k-120k                  1732  
    120k-160k                  400  
    160k-200k                  406  


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/1543974424.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      print(data.groupby('income_bin').agg(Mean_credit_limit=('credit_limit','mean'),Median_credit_limit=('credit_limit','median'),Max_credit_limit=('credit_limit','max'),count_credit_limit=('credit_limit','count')).round())


## Distribution of Customer Transaction amount, Credit limit and Utilization ratios:
+ Credit cards are highly utilized by customers with incomes between 0–40k and $40k–80k, with their transaction amounts closely aligning with the utilization ratios
+ The customer income bins and their corresponding utilization ratios are as follows:
 + 0–40k: 100.03
 + 40k–80k: 55.04
 + 80k–120k: 29.45
 + 120k–160k: 25.70
 + 160k–200k: 23.03


```python
#The utilization of credit card is very high with income bins 0-40k & 40k-80k
grouped_trans_amount = data.groupby('income_bin').agg(total_trans_amount =('total_trans_amount','sum'),total_credit = ('credit_limit','sum'),total_trans_count =('total_trans_count','count'))

grouped_trans_amount['avg_utilization_ratio'] = grouped_trans_amount['total_trans_amount'] / grouped_trans_amount['total_credit']* 100

print(grouped_trans_amount)
```

                total_trans_amount  total_credit  total_trans_count  \
    income_bin                                                        
    0-40k                 17362727    17356418.9               3981   
    40k-80k               15847252    28788968.7               3601   
    80k-120k               7648101    25968957.7               1732   
    120k-160k              1911312     7435078.5                400   
    160k-200k              1792083     7778636.3                406   
    
                avg_utilization_ratio  
    income_bin                         
    0-40k                  100.036344  
    40k-80k                 55.046265  
    80k-120k                29.450936  
    120k-160k               25.706682  
    160k-200k               23.038524  


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/1908500781.py:2: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped_trans_amount = data.groupby('income_bin').agg(total_trans_amount =('total_trans_amount','sum'),total_credit = ('credit_limit','sum'),total_trans_count =('total_trans_count','count'))


## Customer credit limit:
+ Out of 10127 customers,8863 of them falls under the 0-10K & 10K-20K credit limit whose median incomes are 40K and 72k respectively


```python
# Using bins to segment the credit limit
bins = [0, 10000, 20000, 30000, 40000]
bin_labels = ['0-10k', '10k-20k', '20k-30k', '30k-40k']

# Create a new column 'credit_limit' that categorizes income
data['credit_limit_bin'] = pd.cut(data['credit_limit'], bins=bins, labels=bin_labels, right=False)
print(data['credit_limit_bin'].value_counts())
```

    credit_limit_bin
    0-10k      7373
    10k-20k    1490
    30k-40k     667
    20k-30k     597
    Name: count, dtype: int64



```python
grouped_credit_limit_bin = data.groupby('credit_limit_bin').agg(median_income =('estimated_income','median')).round()
print(grouped_credit_limit_bin)
```

                      median_income
    credit_limit_bin               
    0-10k                   40000.0
    10k-20k                 72000.0
    20k-30k                 91000.0
    30k-40k                106000.0


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/2824346797.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped_credit_limit_bin = data.groupby('credit_limit_bin').agg(median_income =('estimated_income','median')).round()


### Utilization ratio based on credit limit
+ The average of Utilization ratio is higher for females compared to males
+ Utilization ratio is higher for lower income range (0-10k,10k-20k)


```python
# Grouping data by gender
print(data.groupby('gender').agg(mean_utilization_ratio = ('avg_utilization_ratio','mean')).round(3))
```

            mean_utilization_ratio
    gender                        
    F                        0.342
    M                        0.200



```python
# Grouping data by credit_limit
print(data.groupby('credit_limit_bin').agg(median_utilization_ratio = ('avg_utilization_ratio','median'),max_utilization_ratio= ('avg_utilization_ratio','max')))
```

                      median_utilization_ratio  max_utilization_ratio
    credit_limit_bin                                                 
    0-10k                               0.3270                  0.999
    10k-20k                             0.0885                  0.250
    20k-30k                             0.0540                  0.124
    30k-40k                             0.0400                  0.083


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/3648060775.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      print(data.groupby('credit_limit_bin').agg(median_utilization_ratio = ('avg_utilization_ratio','median'),max_utilization_ratio= ('avg_utilization_ratio','max')))


### Correlation Analysis with actual data: 


```python
# Importing seaborn library to plot the correlation between the outcome and the numerical predictors
import seaborn as sns
correlation = data[data.select_dtypes(include=['int64','float64']).columns.values].corr()

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, cmap='Blues',annot=True,cbar=True,fmt='.2f')

# Add titles and labels
plt.title('Heatmap correlation values')
plt.xlabel('features')
plt.ylabel('features')

# Show the plot
plt.show()
```


    
![png](output_48_0.png)
    


### Hot encoding: Replacing categorical columns with dummy variables
+ To incorporate the categorical columns into the analysis, I am using one-hot encoding to transform them into dummy variables


```python
## Hot encoding - replacing categorical columns with dummy variables
dat = data.copy()
dat = pd.get_dummies(dat, columns=['marital_status'],dtype=int, drop_first=True)

### Replacing categorical columns with binary values
dat['gender'] = dat['gender'].replace({'F':0,'M':1}) #Placeholders
dat['education_level']= dat['education_level'].replace({'Uneducated':0,'High School':1,'College':2,'Graduate':3,'Post-Graduate':4,'Doctorate':5})
```

    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/4014277735.py:6: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      dat['gender'] = dat['gender'].replace({'F':0,'M':1}) #Placeholders
    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_2007/4014277735.py:7: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      dat['education_level']= dat['education_level'].replace({'Uneducated':0,'High School':1,'College':2,'Graduate':3,'Post-Graduate':4,'Doctorate':5})


### Dropping the customer id,income bin and credit limit bin columns for analysis


```python
dat = dat.drop(columns = ['customer_id','income_bin','credit_limit_bin'])
```


```python
# Current columns in the data
print(dat.columns)
```

    Index(['age', 'gender', 'dependent_count', 'education_level',
           'estimated_income', 'months_on_book', 'total_relationship_count',
           'months_inactive_12_mon', 'credit_limit', 'total_trans_amount',
           'total_trans_count', 'avg_utilization_ratio', 'marital_status_Married',
           'marital_status_Single', 'marital_status_Unknown'],
          dtype='object')


### Correlation analysis including categorical columns: 
+ Gender shows a strong correlation with both credit limit (0.42) and income (0.60). The average income of males ($90K) is higher than that of females (40K)
+ Credit limit has a strong correlation with income (0.52); higher income is associated with higher credit limits
+ Credit limit is also strongly correlated with the utilization ratio (0.48). The higher the credit limit, the greater the usage
+ Age is correlated with the number of months on the books
+ Total transaction count is correlated with total transaction amount(0.81). A higher number of transactions leads to a higher total transaction amount


```python
# Replacing category values with binary values : Gender '1' : Male, '0': Female
print(dat.groupby('gender').agg(credit_limit_mean = ('credit_limit','mean'),income_mean = ('estimated_income','mean')))
```

            credit_limit_mean   income_mean
    gender                                 
    0             5023.854274  39725.270623
    1            12685.674963  87191.864122



```python
# Importing seaborn library to plot the correlation between the numerical features
import seaborn as sns

correlation = abs(dat.corr())

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation[correlation > 0.2], cmap='Blues',annot=True,cbar=True,fmt='.2f')

# Add titles and labels
plt.title('Heatmap correlation values')
plt.xlabel('features')
plt.ylabel('features')

# Show the plot
plt.show()
```


    
![png](output_56_0.png)
    


### Scaling the features:


```python
# Using scaling to standardize the features
from sklearn.preprocessing import StandardScaler

def scaler(df):
    scal = StandardScaler()
    scal.fit(df)
    scal_x = scal.transform(df)
    return scal_x

scaled_data = pd.DataFrame(scaler(dat[['age', 'gender', 'dependent_count', 'education_level',
       'estimated_income', 'months_on_book', 'total_relationship_count',
       'months_inactive_12_mon', 'credit_limit', 'total_trans_amount',
       'total_trans_count', 'avg_utilization_ratio', 'marital_status_Married',
       'marital_status_Single', 'marital_status_Unknown']]), columns=['age', 'gender', 'dependent_count', 'education_level',
       'estimated_income', 'months_on_book', 'total_relationship_count',
       'months_inactive_12_mon', 'credit_limit', 'total_trans_amount',
       'total_trans_count', 'avg_utilization_ratio', 'marital_status_Married',
       'marital_status_Single', 'marital_status_Unknown'])  
```

### Inertia: How far the datapoints are from their respective centroids
+ Inertia is inversely proportional to the number of clusters, meaning it decreases as the number of clusters increases.
+ The optimal number of clusters for data segmentation is determined by the elbow curve.
+ A sharp elbow is observed at Cluster 6, so we are choosing 6 clusters for the segmentation


```python
### Inertia 
def plot_elbow_curve(df,k=10):
    inertias =[]
    cluster =[]
    for i in range(1,k+1):
        model = KMeans(n_clusters=i,random_state = 417)
        clusters = model.fit_predict(df)
        inertias.append(model.inertia_)
        cluster.append(i)

    plt.figure(figsize =(6,4))
    plt.plot(cluster,inertias,marker ='o')
    plt.title('Inertia Vs Cluster')
    plt.show()

    return inertias,cluster

inertias,cluster = plot_elbow_curve(scaled_data)
print(inertias,cluster)
```


    
![png](output_60_0.png)
    


    [151904.99999999977, 134792.16650439642, 125630.66787351093, 116869.80936940908, 107894.94539459074, 103471.98967655396, 94491.44236766097, 91000.99609601361, 88529.87974468282, 86270.75232069491] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



```python
### Formula: Relationship between the inertia and cluster
percent_decrease = [(1 - (inertias[i] / inertias[i - 1])) * 100 for i in range(1, 10)]
print(percent_decrease)
```

    [11.265484016723203, 6.796758942654635, 6.973503088372157, 7.679369054543461, 4.099316888164928, 8.6792061667757, 3.6939284491671054, 2.7154827500168843, 2.5518247968970065]


### Percentage decrease of Inertia compared to its successor


```python
ax = plt.bar(range(2, 11), percent_decrease)
plt.bar_label(ax, fmt='%.1f%%')
plt.title("Percentual decrease in Inertia from Previous K")
plt.xticks(ticks=range(2, 11), labels=range(2, 11))
plt.show()
```


    
![png](output_63_0.png)
    


### KMeans clustering on Actual data using the scaled dataframe


```python
# Invoking the KMeans model
model = KMeans(n_clusters = 6,random_state=700)
data['cluster'] = model.fit_predict(scaled_data)

# Replacing the cluster values accordingly
mapi = {'0':'1','1':'2','2':'3','3':'4','4':'5','5':'6'}
data['cluster'] = data['cluster'].astype(str).replace(mapi)
print(data['cluster'].value_counts())
```

    cluster
    5    2421
    1    2112
    4    1936
    6    1382
    3    1313
    2     963
    Name: count, dtype: int64



```python
# clusters on actual dataframe
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting clusters using features that have a high correlation with each other
fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(12,10))
sns.scatterplot(data= data,x='estimated_income',y='credit_limit',ax=ax1,hue='cluster',palette='deep')
ax1.legend(loc='upper right')
ax1.set_yticks(range(1000, 40000, 3000)) 
ax1.set_xticks(range(20000, 200000, 20000))
ax1.tick_params(axis='x', labelrotation=90)
sns.scatterplot(data= data,x='estimated_income',y='avg_utilization_ratio',ax=ax2,hue='cluster',palette='deep')
sns.scatterplot(data= data,x='total_trans_count',y='total_trans_amount',ax=ax3,hue='cluster',palette='deep')
ax3.set_yticks(range(0, 20000, 2000))
sns.scatterplot(data= data,x='age',y='months_on_book',ax=ax4,hue='cluster',palette='deep')
plt.tight_layout()
plt.show()
```


    
![png](output_66_0.png)
    


### Cluster's distribution and its characteristics:

+ Cluster1:
    + Cluster 1 consists predominantly of married females(84%). Their income ranges from 27,000 to 48,000, while their credit limits vary between 2,106 and 4,402. The utilization ratio for this cluster ranges from 9.9% to 64.5%, with an average of 39.69% highest of all the clusters. Cluster 1 represents 21% of the total transaction count, and the total transaction amount for this cluster accounts for 17% of the overall transactions — the medium among all clusters. They exhibit a trend of making many smaller purchases, may not be paying back on time and maximizing the credit available to them.
     
+ Cluster2:
    + Cluster 2 consists of 64% of customers being Male. Their income ranges from 37,000 to 89,000, while their credit limits fall between 5,178 and $23,231. The utilization ratio for this cluster ranges from 4.6% to 23.4%, with an average of 16.76%. Cluster 2 accounts for 10% of the total transaction count and 29% of the total transaction amount.Additionally, they exhibit a trend of making few bulk purchases, which could further be addressed with targeted marketing strategies.

+ Cluster3:
    +  Cluster 3 consists 66% of customers being Female. Their income ranges from 30,000 to 57,000, and their credit limits fall between 2,214 and $5,579. The utilization ratio for this cluster ranges from 1.9% to 60.5%, with an average of 34.66%. Cluster 3 accounts for 13% of the total transaction count and 9% of the total transaction amount. Their utilization rate is higher with low transaction amount. They exhibit a trend of making many smaller purchases and may not be paying back on time.

+ Cluster4:
  + Cluster 4 consists consists of customers predominantly Married Male(96%). Their income is concentrated between 70,000 and 116,000, and the average income in this cluster is 99,167$ and their credit limits between 5,773 and 21,336. They have a utilization ratio, ranging from 0.00 to 17.3%, with a mean of 12.22%. The transaction count is low, accounting for 19% of the total transactions, and they represent 13% of the total transaction amount. Given their higher income & credit limit, the company could encourage more spending by tailoring products to these customers.


+ Cluster5:
    + Cluster 5 consists of customers predominantly single female(91%). Their income is concentrated between 27,000 and 45,000 and credit limits between 2,077 and 4,871$.They have a utilization ratio, ranging from 7.0 to 63.2%, with a mean of 37.61%. The transaction count is 24% of the total transaction count and 21% of the total transaction amount.Their utilization rate is second highest. Customers in Cluster 5 appear to be maximizing the credit available to them and may not be paying back on time. they have similar characteristics to cluster 1.


+ Cluster6:
    + Cluster 6 consists predominantly of Single Males(94%). Their income is concentrated between 67,000 and 114,000, and their credit limits between 5,635 and 22,344$. They have a utilization ratio, ranging from 0.00 to 17.67%, with a mean of 13.14%.
The transaction count is 14% of the total transaction count and 11% of the total transaction amount. They have similar characteristics to Cluster 4.

### Numerical features: Distribution in clusters


```python
income_distribution = data.groupby('cluster').agg(mean_income = ('estimated_income','mean')).round()
creditlimit_distribution = data.groupby('cluster').agg(mean_credit_limit = ('credit_limit','mean')).round()
avg_utilization_ratio_distribution = data.groupby('cluster').agg(mean_utilization_ratio = ('avg_utilization_ratio','mean'))
avg_utilization_ratio_distribution['mean_utilization_ratio '] = avg_utilization_ratio_distribution['mean_utilization_ratio'] * 100
total_trans_count_distribution = data.groupby('cluster').agg(Total_trans_count = ('total_trans_count','count'))
total_trans_amount_distribution = data.groupby('cluster').agg(Total_trans_amount = ('total_trans_amount','sum'))


# Charts:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(5,1,figsize=(12,10))
income_distribution.plot.bar(stacked=True, ax=ax[0], alpha=0.6)
ax[0].set_title(f' Mean income per Cluster', alpha=0.5)
ax[0].legend(loc='upper right',fontsize=8, markerscale=2)
ax[0].xaxis.grid(False)
ax[0].tick_params(axis='x', labelrotation=0)

creditlimit_distribution.plot.bar(stacked=True, ax=ax[1], alpha=0.6,color = 'Grey')
ax[1].set_title(f'Mean credit_limit per Cluster', alpha=0.5)
ax[1].legend(loc='upper right',fontsize=8, markerscale=2)
ax[1].xaxis.grid(False)
ax[1].tick_params(axis='x', labelrotation=0)

avg_utilization_ratio_distribution.plot.bar(stacked=True, ax=ax[2], alpha=0.6, color = 'Orange')
ax[2].set_title(f'% Mean Utilization ratio per Cluster', alpha=0.5)
ax[2].legend(loc='upper right',fontsize=8, markerscale=2)
ax[2].xaxis.grid(False)
ax[2].tick_params(axis='x', labelrotation=0)

total_trans_count_distribution.plot.bar(stacked=True, ax=ax[3], alpha=0.6)
ax[3].set_title(f'Total trans count per Cluster', alpha=0.5)
ax[3].legend(loc='upper right',fontsize=8, markerscale=2)
ax[3].xaxis.grid(False)
ax[3].tick_params(axis='x', labelrotation=0)

total_trans_amount_distribution.plot.bar(stacked=True, ax=ax[4], alpha=0.6, color ='grey')
ax[4].set_title(f'Total_trans_amount per Cluster', alpha=0.5)
ax[4].legend(loc='upper right',fontsize=8, markerscale=2)
ax[4].xaxis.grid(False)
ax[4].tick_params(axis='x', labelrotation=0)

plt.show()
```


    
![png](output_69_0.png)
    


### Categorical features: Distribution in clusters


```python
### Percental distribution variable per cluster ( Normalized - range between 0-1)
gender_distribution = pd.crosstab(index=data['cluster'], columns=data['gender'],values=data['gender'], aggfunc='size',normalize= 'index')
marital_status_distribution = pd.crosstab(index=data['cluster'], columns=data['marital_status'],values=data['marital_status'], aggfunc='size',normalize= 'index')
dependent_count_distribution = pd.crosstab(index=data['cluster'], columns=data['dependent_count'],values=data['dependent_count'], aggfunc='size',normalize= 'index')
income_bin_distribution = pd.crosstab(index=data['cluster'], columns=data['income_bin'],values=data['income_bin'], aggfunc='size',normalize= 'index')
credit_limit_bin_distribution = pd.crosstab(index=data['cluster'], columns=data['credit_limit_bin'],values=data['credit_limit_bin'], aggfunc='size',normalize= 'index')

# Charts:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(5,1,figsize=(12,10))
gender_distribution.plot.bar(stacked=True, ax=ax[0], alpha=0.6)
ax[0].set_title(f'% Gender per Cluster', alpha=0.5)
ax[0].set_ylim(0, 1.5)
ax[0].legend(loc='upper right',fontsize=8, markerscale=2)
ax[0].xaxis.grid(False)
ax[0].tick_params(axis='x', labelrotation=0)

marital_status_distribution.plot.bar(stacked=True, ax=ax[1], alpha=0.6)
ax[1].set_title(f'% marital_status per Cluster', alpha=0.5)
ax[1].set_ylim(0, 1.2)
ax[1].legend(loc='upper right',fontsize=8, markerscale=2)
ax[1].xaxis.grid(False)
ax[1].tick_params(axis='x', labelrotation=0)

dependent_count_distribution.plot.bar(stacked=True, ax=ax[2], alpha=0.6)
ax[2].set_title(f'% dependent_count per Cluster', alpha=0.5)
ax[2].set_ylim(0, 1.2)
ax[2].legend(loc='upper right',fontsize=8, markerscale=2)
ax[2].xaxis.grid(False)
ax[2].tick_params(axis='x', labelrotation=0)

income_bin_distribution.plot.bar(stacked=True, ax=ax[3], alpha=0.6)
ax[3].set_title(f'% income_bin per Cluster', alpha=0.5)
ax[3].set_ylim(0, 1.2)
ax[3].legend(loc='upper right',fontsize=8, markerscale=2)
ax[3].xaxis.grid(False)
ax[3].tick_params(axis='x', labelrotation=0)

credit_limit_bin_distribution.plot.bar(stacked=True, ax=ax[4], alpha=0.6)
ax[4].set_title(f'% credit_limit_bin per Cluster', alpha=0.5)
ax[4].set_ylim(0, 1.2)
ax[4].legend(loc='upper right',fontsize=8, markerscale=2)
ax[4].xaxis.grid(False)
ax[4].tick_params(axis='x', labelrotation=0)

plt.tight_layout()
plt.show()
```


    
![png](output_71_0.png)
    

