#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:21:32 2017

@author: streiff
"""


'''0-Import from my Mac'''
import pandas as pd

df_test=pd.read_csv(r'/Users/streiff/Downloads/dengue_driven_data/dengue_features_test.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
df_test.head() #24 columns
df_test.info()# 416 entries non-nul object
df_train=pd.read_csv('/Users/streiff/Downloads/dengue_driven_data/dengue_features_train.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
#24 columns, 1456 entries
dl_train=pd.read_csv(r'/Users/streiff/Downloads/dengue_driven_data/dengue_labels_train.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
#4 col: city, year, weakofyear, total cases,  1456 entries
sf=pd.read_csv(r'/Users/streiff/Downloads/dengue_driven_data/submission_format.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
# 4 cols, 416 entries similar to dl_train

'''0-Import from my Linux'''
df_test=pd.read_csv(r'/home/erika/dengue driven data/dengue_features_test.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
df_train=pd.read_csv(r'/home/erika/dengue driven data/dengue_features_train.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
dl_train=pd.read_csv(r'/home/erika/dengue driven data/dengue_labels_train.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
sf=pd.read_csv(r'/home/erika/dengue driven data/submission_format.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)

#dl_train is from sj 1990 till iq 2010
#sf is from sj 2008 till iq 2013 (need to predict data from 2011 to 2013)

'''1-Explore'''

df_test.city.value_counts(dropna=False) #sj    260    iq    156
df_train.city.value_counts(dropna=False) #sj    936    iq    520 = dl.train
df_train['year'][100] # 1992
#set_mutiindex
df_train_multiindex=df_train.reset_index().set_index(['city','year','weekofyear'])
df_test_multiindex=df_test.reset_index().set_index(['city','year','weekofyear'])
dl_train_multiindex=dl_train.reset_index().set_index(['city','year','weekofyear'])


'''2-cleaning'''
df_train_multiindex.drop(df_train_multiindex.columns[[0,1]], axis=1, inplace=True) #drop col 0 and 1 (index and weekstartdate)
df_train_multiindex.sort_index() #incomplete week: sj 2008, sj1990, iq 2000, 2010

df_test_multiindex.drop(df_test_multiindex.columns[[0,1]], axis=1, inplace=True) #drop col 0 and 1 (index and weekstartdate)
df_test_multiindex.sort_index() #incomplete week: sj 2008, sj1990, iq 2000, 2010

del dl_train_multiindex['index']
dl_train_multiindex.sort_index()


# join df_train & dl_train if multiindex is matched (using index)
train = dl_train_multiindex.reset_index().join(df_train_multiindex,on=['city','year','weekofyear']).set_index(dl_train_multiindex.index.names)

#shift the total case col up by 1
train['total_cases']=train['total_cases'].shift(-2) # shifted 3 & 1 weeks performed worse than 2, shifted 2 performs slightly worse than unshifted 
train=train[:-2]


#heat map

from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

sns.set(style="white")


# Compute the correlation matrix
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


y_train=train.iloc[:, [0]].values #type:numpy.ndarray
x_train=train.iloc[:, [1,2,3,4,6,7,8,9,10,12,14,15,16,17,19,20]].values #type:numpy.ndarray
x_test=df_test_multiindex.iloc[:, [0,1,2,3,5,6,7,8,9,11,13,14,15,16,18,19]].values

'''pipeline: SVC: worse than linear regression but I will try we the unshifted data'''
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)), # try to change the strategy
         ('scaler', StandardScaler()),
         ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)


# Fit the pipeline to the train set
pipeline.fit(x_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(x_test)

# Compute metrics
print(classification_report(y_test, y_pred))


------

'''Linear Regression'''

'''imputation''' #to avoid ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
imputer = Imputer()
x_train = imputer.fit_transform(x_train)
y_train = imputer.fit_transform(y_train)
x_test = imputer.fit_transform(x_test)

'''Scaling the variables'''

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)


'''ML part'''
# Import
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# PolynomialFeatures (prepreprocessing)
poly = PolynomialFeatures(degree=2)

X_train = poly.fit_transform(x_train)# X_train.shape = (1456, 231)), x_train.shape (1456, 20)
X_test = poly.fit_transform(x_test) # X_test.shape:(416, 231),x_test.shape (416, 20)

# Instantiate
lg = LinearRegression()

# Fit
lg.fit(X_train,y_train)
import numpy as np


# Obtain coefficients
coeff=lg.coef_

# Predict
y_pred=lg.predict(X_test) #numpy array (416,1)
np.savetxt('y_pred.csv', y_pred, delimiter=',')

'''submission'''
df_test_multiindex['total_cases'] = y_pred.tolist()
sf=df_test_multiindex['total_cases'].reset_index()
sf['total_cases'] = sf['total_cases'].str[0] #remove square bracket

sf['total_cases'] = sf['total_cases'].fillna(0.0).astype(int)
sf.to_csv('submission_erika3.csv', sep='\t')


''' Measure the accuracy: impossible bcos the y_test is hidden'''

from sklearn.metrics import accuracy_score
sklearn.metrics.accuracy_score(y_train, y_pred, normalize=True, sample_weight=None)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_pred = scaler_y.transform(y_pred)
print(confusion_matrix(y_test,y_pred))

'''Find tunning'''


------
label = pd.merge(sf, dl_train, how='left', left_on=['city','year','weekofyear'],right_on=['city','year','weekofyear'])
label=sf.reset_index().join(dl_train,on=['city','year','weekofyear'],how='left').set_index(dl_train_multiindex.index.names)
label1=label.dropna(axis=0,how='any')


sf1=sf.set_index(['city','year','weekofyear'])