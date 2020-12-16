# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:48:50 2020

@author: rohir
"""

# Importing required libraries
import numpy as np
import pandas as pd
import os as os
import seaborn as sns

sns.set(style="ticks", color_codes=True)



#changing the working directory
os.chdir("C:\\Users\\rohir\\OneDrive\\Documents\\Summer 2020\\Python\\project")

#reading the csv file
nyc = pd.read_csv("AB_NYC_2019.csv")

#Data type info
nyc.dtypes

#Dropping unnecessary columns
nyc.drop(["id","host_name","latitude","longitude"], axis=1, inplace=True)

nyc.head()
nyc.info()

nyc.isnull().sum()

#Imputing missing values
nyc.fillna({'name':"NoName"}, inplace=True)
nyc.fillna({'last_review':"NotReviewed"}, inplace=True)
nyc.fillna({'reviews_per_month':0}, inplace=True)

nyc.isnull().sum()

#Changing datatype for host id
nyc.host_id = nyc.host_id.astype(str)
nyc.dtypes

#Removing outliers in price column
nyc["price"].describe()

m3 = nyc["price"].mean() + (3 * nyc["price"].std())
m3

sum(nyc["price"]>m3)

l3 = nyc["price"].mean() - (3 * nyc["price"].std())
l3

sum(nyc["price"]<l3)

nyc_cleaned=nyc[nyc["price"]<m3]

nyc_cleaned["price"].describe()

#final cleaned dataset is nyc_cleaned
