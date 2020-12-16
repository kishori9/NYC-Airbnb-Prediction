# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:28:34 2020

@author: Kishori
"""

# Importing required libraries
import numpy as np
import pandas as pd
import os as os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn import svm
from sklearn import feature_importances_
sns.set(style="ticks", color_codes=True)

#reading the csv file
nyc = pd.read_csv("AB_NYC_2019.csv")

#Data type info
nyc.dtypes

#Dropping unnecessary columns
nyc.drop(["id","host_name"], axis=1, inplace=True)

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

df = nyc_cleaned
df.info()

df.isnull().sum()

df['neighbourhood_group']= df['neighbourhood_group'].astype("category").cat.codes
df['neighbourhood'] = df['neighbourhood'].astype("category").cat.codes
df['room_type'] = df['room_type'].astype("category").cat.codes
df.info()

#Imputing reviews_per_month
mean = df['reviews_per_month'].mean()
df['reviews_per_month'].fillna(mean, inplace=True)
df.isnull().sum()


# price conversion
df['price_log'] = np.log(df.price+1)

#correlation matrix
corrmat = df.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#Clean dataset
df.to_csv('clean.csv')

#Set random seed
import random
random.seed(10)

# Modelling code
X = df[["neighbourhood","neighbourhood_group","latitude","longitude","room_type","minimum_nights","availability_365","number_of_reviews","calculated_host_listings_count","reviews_per_month"]]
y = df["price_log"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state = 0)
Poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_train = Poly.fit_transform(X_train)
X_test = Poly.fit_transform(X_test)

#Standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

import matplotlib.pyplot as plt
# Linear regression model
print("Linear Regression metrics:")
lr = LinearRegression()
lr.fit(X_train_std,y_train)
train = lr.predict(X_train_std)
prediction = lr.predict(X_test_std)
RMSE_test = np.sqrt(mean_squared_error(y_test,prediction))
RMSE_train = np.sqrt(mean_squared_error(y_train,train))
print(RMSE_train)
print(RMSE_test)
r_2_test = sklearn.metrics.r2_score(y_test,prediction)
r_2_train = sklearn.metrics.r2_score(y_train,train)
print(r_2_train)
print(r_2_test)

lr.summary()

plt.scatter(y_train, train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#Ridge Model
print("Ridge metrics:")
ridge_model = Ridge(alpha = 0.01, normalize = True)
ridge_model.fit(X_train_std, y_train)             
ridge_train = ridge_model.predict(X_train_std)
ridge_prediction = ridge_model.predict(X_test_std)
ridge_RMSE_test = np.sqrt(mean_squared_error(y_test,ridge_prediction))
ridge_RMSE_train = np.sqrt(mean_squared_error(y_train,ridge_train))
print(ridge_RMSE_train)
print(ridge_RMSE_test)
ridge_r_2_test = sklearn.metrics.r2_score(y_test,ridge_prediction)
ridge_r_2_train = sklearn.metrics.r2_score(y_train,ridge_train)
print(ridge_r_2_train)
print(ridge_r_2_test)

plt.scatter(y_train, ridge_train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, ridge_prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#Lasso Model
print("Lasso metrics:")
lasso_model = Lasso(alpha = 0.0001, normalize = False)
lasso_model.fit(X_train_std, y_train)             
lasso_train = lasso_model.predict(X_train_std)
lasso_prediction = lasso_model.predict(X_test_std)
lasso_RMSE_test = np.sqrt(mean_squared_error(y_test,lasso_prediction))
lasso_RMSE_train = np.sqrt(mean_squared_error(y_train,lasso_train))
print(lasso_RMSE_train)
print(lasso_RMSE_test)
lasso_r_2_test = sklearn.metrics.r2_score(y_test,lasso_prediction)
lasso_r_2_train = sklearn.metrics.r2_score(y_train,lasso_train)
print(lasso_r_2_train)
print(lasso_r_2_test)

#Plots
#training set
plt.scatter(y_train, lasso_train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, lasso_prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#ElasticNet Model
print("ElasticNet metrics:")
ENet_model = ElasticNet(alpha = 0.01, normalize = False)
ENet_model.fit(X_train_std, y_train)             
ENet_train = ENet_model.predict(X_train_std)
ENet_prediction = ENet_model.predict(X_test_std)
ENet_RMSE_test = np.sqrt(mean_squared_error(y_test,ENet_prediction))
ENet_RMSE_train = np.sqrt(mean_squared_error(y_train,ENet_train))
print(ENet_RMSE_train)
print(ENet_RMSE_test)
ENet_r_2_test = sklearn.metrics.r2_score(y_test,ENet_prediction)
ENet_r_2_train = sklearn.metrics.r2_score(y_train,ENet_train)
print(ENet_r_2_train)
print(ENet_r_2_test)

#Plots
#training set
plt.scatter(y_train, train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

# KNN
print("KNN metrics:")
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    knn = neighbors.KNeighborsRegressor(n_neighbors = K)
    knn.fit(X_train_std, y_train)
    knn_train =knn.predict(X_train_std)
    knn_prediction = knn.predict(X_test_std) 
    knn_RMSE_test = np.sqrt(mean_squared_error(y_test,knn_prediction)) #calculate rmse
    knn_RMSE_train = np.sqrt(mean_squared_error(y_train,knn_train))
    print(K)
    print(knn_RMSE_train)
    print(knn_RMSE_test)
    knn_r_2_test = sklearn.metrics.r2_score(y_test,knn_prediction)
    knn_r_2_train = sklearn.metrics.r2_score(y_train,knn_train)
    print(knn_r_2_train)
    print(knn_r_2_test)

#Optimum r-square is at K = 19
    
print("KNN metrics for optimum K: ")
knn = neighbors.KNeighborsRegressor(n_neighbors = 19)
knn.fit(X_train_std, y_train)
knn_train =knn.predict(X_train_std)
knn_prediction = knn.predict(X_test_std) 
knn_RMSE_test = np.sqrt(mean_squared_error(y_test,knn_prediction)) #calculate rmse
knn_RMSE_train = np.sqrt(mean_squared_error(y_train,knn_train))
print("For K = 19 (Optimum Value of K)")
print(knn_RMSE_train)
print(knn_RMSE_test)
knn_r_2_test = sklearn.metrics.r2_score(y_test,knn_prediction)
knn_r_2_train = sklearn.metrics.r2_score(y_train,knn_train)
print(knn_r_2_train)
print(knn_r_2_test)

#Plots
#training set
plt.scatter(y_train, knn_train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, knn_prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

   
#Decision Tree Regressor
print("Decision Tree Regressor metrics:")
dt = DecisionTreeRegressor(random_state=0, splitter='best')
dt.fit(X_train_std, y_train)
dt_train = dt.predict(X_train_std)
dt_prediction = dt.predict(X_test_std)
dt_RMSE_test = np.sqrt(mean_squared_error(y_test,dt_prediction))
dt_RMSE_train = np.sqrt(mean_squared_error(y_train,dt_train))
print(dt_RMSE_train)
print(dt_RMSE_test)
dt_r_2_test = sklearn.metrics.r2_score(y_test,dt_prediction)
dt_r_2_train = sklearn.metrics.r2_score(y_train,dt_train)
print(dt_r_2_train)
print(dt_r_2_test)

#Plots
#training set
plt.scatter(y_train, dt_train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, dt_prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


#Random Forest Regressor
print("Random Forest Regressor metrics:")
regr = RandomForestRegressor(max_depth=2, random_state=0) 
regr.fit(X_train_std, y_train)
regr_train = regr.predict(X_train_std)
regr_prediction = regr.predict(X_test_std)
regr_RMSE_test = np.sqrt(mean_squared_error(y_test,regr_prediction))
regr_RMSE_train = np.sqrt(mean_squared_error(y_train,regr_train))
print(regr_RMSE_train)
print(regr_RMSE_test)
regr_r_2_test = sklearn.metrics.r2_score(y_test,regr_prediction)
regr_r_2_train = sklearn.metrics.r2_score(y_train,regr_train)
print(regr_r_2_train)
print(regr_r_2_test)

#Plots
#training set
plt.scatter(y_train, regr_train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, regr_prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


#SVM
print("SVM metrics:")
svm = svm.SVR()
svm.fit(X_train_std, y_train)
svm_train = svm.predict(X_train_std)
svm_prediction = svm.predict(X_test_std)
svm_RMSE_test = np.sqrt(mean_squared_error(y_test,svm_prediction))
svm_RMSE_train = np.sqrt(mean_squared_error(y_train,svm_train))
print(svm_RMSE_train)
print(svm_RMSE_test)
svm_r_2_test = sklearn.metrics.r2_score(y_test,svm_prediction)
svm_r_2_train = sklearn.metrics.r2_score(y_train,svm_train)
print(svm_r_2_train)
print(svm_r_2_test)

#Plots
#training set
plt.scatter(y_train, svm_train, color = 'blue')
plt.title('Actual Vs Predicted Price (Training set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#test set
plt.scatter(y_test, svm_prediction, color = 'red')
plt.title('Actual Vs Predicted Price (Test set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#Feature importance
def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()
cv = CountVectorizer()
cv.fit(df)
print cv.get_feature_names()
plot_coefficients(svm, cv.get_feature_names())

feat_labels = df.columns[[2, 3, 4, 5, 6, 8, 9, 11, 12, 13]]
print(feat_labels)
importances = svm.coef_
indices = np.argsort(importances)[::-1]
for f in range(X_train_std.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


