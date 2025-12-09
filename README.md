# Sample
Kaggle Challenge

Titanic Data

## Step 1 - Import Necessary Library

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Step 2 - Read Dataset

# List of Sheets Available in Excel 
file_path = r'C:\Users\nhapinder\Desktop\train(2).xlsx'
sheet = pd.ExcelFile(file_path)
myWorkSheets = sheet.sheet_names
myWorkSheets

sheet_name = "train"
df = pd.read_excel(file_path, sheet_name)
df.head()

# Step 3 - Sanity Check of the Data

# Shape
df.shape

# info()
df.info()

# Describe 
df.describe()

# Finding Missing Value 
df.isnull().sum()

# Finding Duplicates 
df.duplicated().sum()

# 1 means Survived
(df['Survived'] == 1).sum()

# Identifying Garbage Values
for i in df.select_dtypes(include = "object").columns:
    print(df[i].value_counts())
    print("***"*10)

## Step 4 - Exploratory Data Analysis (EDA)

# Histogram to Understand the Distribution 
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()

# Heatmap 
import warnings
warnings.filterwarnings("ignore")

sns.heatmap(df.corr(), cmap="YlGnBu")
plt.show()

#  Scatter Plot to Understand the Relationship
for i in ['Fare']:
    sns.scatterplot(data=df, x=i, y='Pclass')
plt.show()

df.select_dtypes(include="number").columns
df.select_dtypes(include="object").columns

n = len(df)
prob_survived = (df['Survived'] == 0).sum() / n
prob_female = (df['Sex'] == 'female').sum() / n

prob_survived
prob_female

# Step 5 - Missing Value Treatments

from sklearn.impute import KNNImputer
impute=KNNImputer()
for i in df.select_dtypes(include="number").columns:
    df[i]=impute.fit_transform(df[[i]])
df.isnull().sum()

# Step 6 - Outlier Treatments 

def wisker(col):
    q1, q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw, uw

  wisker(df['Fare'])
  df.columns

  for i in ['Fare']:
    lw, uw=wisker(df[i])
    df[i]=np.where(df[i]<lw,lw,df[i])
    df[i]=np.where(df[i]>uw,uw,df[i])
  for i in ['Fare']:
    sns.boxplot(df[i])
    plt.show()

  # Step 7 - Encoding Data 

pd.get_dummies(data=df,columns=["Survived","Pclass"],drop_first=True)

# Step 8 - Stratified Shuffle Split 

from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
for train_indices, test_indices in split.split(df, df[["Survived", "Pclass", "Sex"]]):
    strat_train_set = df.loc[train_indices]
    strat_test_set = df.loc[test_indices]
  
    
