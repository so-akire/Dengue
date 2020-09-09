#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:54:02 2017

@author: erika
"""

'''0-Import from my Mac'''
import pandas as pd

df_test=pd.read_csv(r'/Users/streiff/Downloads/dengue driven data/dengue_features_test.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
df_test.head() #24 columns
df_test.info()# 416 entries non-nul object
df_train=pd.read_csv('/Users/streiff/Downloads/dengue_driven_data/dengue_features_train.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
#24 columns, 1456 entries
dl_train=pd.read_csv(r'/home/erika/Downloads/dengue driven data/dengue_labels_train.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
#4 col: city, year, weakofyear, total cases,  1456 entries
sf=pd.read_csv(r'/Users/streiff/Downloads/dengue driven data/submission_format.csv', sep=',', header=0, skipinitialspace=True,low_memory=False)
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
y_train=train.iloc[:, [0]].values #type:numpy.ndarray
x_train=train.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values #type:numpy.ndarray
x_test=df_test_multiindex.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values

'''imputation''' #to avoid ValueError: Input contains NaN, infinity or a value too large for dtype('float64')

imputer = Imputer()
x_train = imputer.fit_transform(x_train)
y_train = imputer.fit_transform(y_train)
x_test = imputer.fit_transform(x_test)

'''Scaling the variables'''
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import preprocessing as pre
scaler_x = pre.StandardScaler()
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
 sf.to_csv('submission_erika1.csv', sep='\t')


''' Measure the accuracy'''

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
dl1=dl_train.set_index(['city','year','weekofyear'])

dl1.loc[pd.IndexSlice['sj':'iq', 2008:2013],18:26]
label = pd.merge(sf, dl_train, how='left', on=index)


sub = sf.reset_index().join(df_train_multiindex,on=['city','year','weekofyear']).set_index(dl_train_multiindex.index.names)












#slice iq and sj
iq=df_train_multiindex.loc['iq']
len(iq) # 520

sj=df_train_multiindex.loc['sj']
len(sj) # 936

df_train.loc[df_train['year'] == '1990'].count()
df_train.groupby('year')['city'].count() 

'''1990:35,
   91 - 99: 52
   2000:78
   2001-2007:104
   2008:69, 2009:53, 2010:26'''
  
#erase iq 2000, 2010
#erase sj 2008, sj1990
#change nan into null

import numpy as np
A[np.isnan(A)] = 0

'''Create Dummy Variable'''

#get dumy of the city. Result iq=0, sj=1
dft_dummy=pd.get_dummies(df_train['city'])

#drop iq. Result col iq 
dft_dummy_dropiq=dft_dummy.drop('iq', axis=1)

dft_dummy_dropiq.head()

# replace col city with qj dummy to df_train
df_train['city']= dft_dummy_dropiq

'''create Matrix t get x:
    
myArray=[[1,2],[3,4]]
DataFrame.as_matrix(columns=None)
matrix_1= df_train.as_matrix(columns=None)'''
#create transpose. column into row and row int col
matrix_2=df_train.transpose()

#create matrix....yes!!!
matrix_3=matrix_2.as_matrix(columns=None)
x=matrix_3

'''Process dl_train to  get y'''

'''PLOTTING'''

# plot dl_train 
import matplotlib.pyplot as plt
dl_train_plot=plt.scatter(dl_train['year'],dl_train['total_cases'])

from bokeh.plotting import figure, show, output_file
p = figure(plot_width=1000, plot_height=400,
           title=None, toolbar_location="below")
p.circle(dl_train['year'],dl_train['total_cases'], size=10)
output_file("scatter.html")
show(p) # less than 100 case each week in the whole year, big numbers shown in 1994,1998,2007

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') #111 means "1x1 grid, first subplot"

ax.scatter(dl_train['year'],dl_train['weekofyear'], dl_train['total_cases'], c='r', marker='o')

ax.set_xlabel('year')
ax.set_ylabel('weekofyear')
ax.set_zlabel('total_cases')

plt.show()

 #heat map
 dl_train_multiindex=dl_train.reset_index().set_index(['city','year','weekofyear'])
 iq3=dl_train_multiindex.loc['iq']
temp1 =iq3.groupby(['year','weekofyear']).total_cases.count()

import matplotlib.pyplot as plt
import numpy as np

# Generate Data
data=temp1.values
test=data.reshape(20,52)
#data = np.random.rand(7,24)
rows = dl_train['year'].unique()
columns = dl_train['weekday'].unique()

plt.pcolor(test,cmap=plt.cm.Reds)

plt.show()
plt.close()
 

#cool graph
from math import pi
from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)
from bokeh.plotting import figure

dl_train['year'] = dl_train['year'].astype(str)
dl_train2 = dl_train.set_index('year')
dl_train2.drop('city', axis=1, inplace=True)


year = list(dl_train2.index)
weekofyear = list(dl_train2.columns)

# reshape to 1D array or rates with a month and year for each row.
dl_train2 = dl_train2.reset_index()


colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=dl_train2.total_cases.min(), high=dl_train2.total_cases.max())
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

source = ColumnDataSource(dl_train2)

p = figure(title="Dengue Case",
           x_range=year, y_range=list(reversed(weekofyear)),
           x_axis_location="above", plot_width=900, plot_height=400,
           tools=TOOLS, toolbar_location='below')

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(x="year", y="weekofyear", width=1, height=1,
       source=source,
       fill_color={'field': 'total_cases', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d%%"),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')

p.select_one(HoverTool).tooltips = [
     ('date', '@weekofyear @year'),
     ('total_cases', '@total_cases%'),
]

show(p)      

sj_case=len(dl_train.reset_index().set_index('city').loc['sj'])

sj_case# 936

iq_case=len(dl_train.reset_index().set_index('city').loc['iq'])

iq_case # 520


np.any(np.isnan(x_train)) #True, so must be cleaned below
import numpy as np
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

y=x_train
nans, x= nan_helper(y)
y[nans]= np.interp(x(nans), x(~nans), y[~nans])
print (y.round(2))
 
np.any(np.isnan(x_train)) #False

train.count()
train.interpolate().count()

'''3-Prepare y_train'''
import numpy as np
# Separate col total_case as y, and create Matrix X of all other col (minus id columns)
y_train=train['total_cases']
y_train=y_train.as_matrix(columns=None).astype(np.float64) #shape 1456,astype changed pandas.core.series.Series into numpy.ndarray
np.all(np.isfinite(y_train)) #True. so it must be cleaned below


'''4-prepare x_train'''
x_train=train.drop('total_cases', axis=1, inplace=True)
#x_train=train.transpose().
x_train=x_train.as_matrix(columns=None).astype(np.float64) #shape 20x1456 type:numpy.ndarray



'''5-prepare x_test: Process df_test to get X_test and interpolate'''
#x_test=df_test_multiindex.transpose()
x_test=df_test_multiindex.as_matrix(columns=None).astype(np.float64) #shape (20, 416), astype changed pandas.core.series.Series into numpy.ndarray
np.all(np.isfinite(x_test)) #True. So must be cleaned


#interpolate NAN
df_train_multiindex.count() #result: many nan, all ndvi &sation have nan + reanalysis_sat_precip_amt_mm
df_train_multiindex.interpolate().count() #all 1456
df_test_multiindex.count() #all have nan 1-12
df_test_multiindex.interpolate().count() # all 416
