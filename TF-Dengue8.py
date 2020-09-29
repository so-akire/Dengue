#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:22:45 2018

@author: erika
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:46:42 2018

@author: erika
"""
"""
Created on Wed Dec  6 19:49:37 2017

@author: streiff
"""

import pandas as pd
import numpy as np

# Point to where you've stored the CSV file on your local machine

dftest =pd.read_csv('dengue_features_test.csv', sep=",", header=0) #24 columns, 416 entries non-nul object
dftrain =pd.read_csv('dengue_features_train.csv', sep=",", header=0)#24 columns, 1456 entries
dltrain =pd.read_csv( 'dengue_labels_train.csv', sep=",", header=0)#4 col: city, year, weakofyear, total cases,  1456 entries, sj 1990 till iq 2010

# sf=target=4 cols, 416 entries similar to ltrain,  sj 2008 till iq 2013 (need to predict data from 2011 to 2013)
#set_mutiindex
dftrain_midx=dftrain.reset_index().set_index(['city','year','weekofyear'])
dftest_midx=dftest.reset_index().set_index(['city','year','weekofyear'])
dltrain_midx=dltrain.reset_index().set_index(['city','year','weekofyear'])


'''2-cleaning'''
#drop col 0 and 1 (index and weekstartdate then sort
dftrain_midx.drop(dftrain_midx.columns[[0,1]], axis=1, inplace=True)
dftrain_midx.sort_index() #incomplete week: sj 2008, sj1990, iq 2000, 2010

dftest_midx.drop(dftest_midx.columns[[0,1]], axis=1, inplace=True) 
dftest_midx.sort_index() #incomplete week: sj 2008, sj1990, iq 2000, 2010

del dltrain_midx['index'] 
dltrain_midx.sort_index()

# join dftrain & dltrain if multiindex is matched (using index)
data = dltrain_midx.reset_index().join(dftrain_midx,on=['city','year','weekofyear']).set_index(dltrain_midx.index.names)

#shift the total case col up by 1
data['total_cases']=data['total_cases'].shift(-2) # shifted 3 & 1 weeks performed worse than 2, shifted 2 performs slightly worse than unshifted 
data=data[:-2] # erase the last 2rows

'''remove outlier with quantile or with 3 times of std: 
q = data['total_cases'].quantile(0.99)
data[data['total_cases'] < q] '''
data= data[np.abs(data.total_cases-data.total_cases.mean())<=(3*data.total_cases.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

#or with lambda
#train_clean=train[train.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]


y=data.iloc[:, [0]].values #type:numpy.ndarray
x=data.iloc[:, np.r_[1:20]].values #type:numpy.ndarray
x_pred=dftest_midx.iloc[::, np.r_[1:20]].values

dataset_size = 1430
import tensorflow as tf

# Model linear regression y = W1x1 + W2x2 + b

W = tf.Variable(tf.zeros([1,19]))
x1 = tf.placeholder(tf.float32, shape=(19,1430))
Wx1 = tf.matmul(W,x1)
b = tf.Variable(tf.zeros([1]), name="b")
y1 = Wx1 +b
y_ = tf.placeholder(tf.float32, [None, 1])
cost = tf.reduce_mean(tf.square(y_ - y1))
train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)

def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

      if dataset_size == batch_size:
        batch_start_idx = 0
      elif dataset_size < batch_size:
        raise ValueError("dataset_size: %d, must be greater than batch_size: %d" % (dataset_size, batch_size))
      else:
        batch_start_idx = (i * batch_size) % dataset_size

      batch_end_idx = batch_start_idx + batch_size

      batch_x = x[batch_start_idx : batch_end_idx]
      batch_y = y[batch_start_idx : batch_end_idx]

      feed = { x1:  batch_x , y_: batch_y }

      sess.run(train_step_ftrl, feed_dict=feed)

      # Print result to screen for every 500 iterations
      if (i + 1) % 500 == 0:
        print("After %d iteration:" % i)
        print("W: %s" % sess.run(W))
        print("b: %f" % sess.run(b))

        print("cost: %f" % sess.run(cost, feed_dict=feed))
        
trainWithMultiplePointsPerEpoch(4000, 100, 10)
trainWithMultiplePointsPerEpoch(5000, train_step_ftrl, len(df[1]))

'''After 3499 iteration:
W1: [[-3.8862844]]
W2: [[-2.405667]]
b: 20.058281
cost: 19.147699 (this is slightly better)

After 4999 iteration:
W1: [[-3.836434]]
W2: [[-2.463515]]
b: 20.271908
cost: 491.811920'''

# Fit the pipeline to x prediction
pipeline.fit(x_pred)
pca_features_pred = pipeline.transform(x_pred)
df=pd.DataFrame(pca_features_pred)
x1_p= df[0].values
x2_p= df[1].values
y_pred=-3.8862844 *x1_p -2.405667*x2_p + 20.058281
y_pred2=-3.836434 *x1_p -2.463515*x2_p + 20.271908

'''submission'''
dftest_midx['total_cases'] = y_pred.tolist()
sf=dftest_midx['total_cases'].reset_index()
#----sf['total_cases'] = sf['total_cases'].str[0] #to remove square bracketnp.savetxt('y_pred.csv', y_pred, delimiter=',') if any

sf['total_cases'] = sf['total_cases'].fillna(0.0).astype(int)
sf.to_csv('submission_erika3.csv', sep='\t')


'''compare to linear model in sklearn'''

import numpy as np
from sklearn import datasets, linear_model
combined = np.vstack((pca1 , pca2)).T
Model = linear_model.LinearRegression()
Model.fit(combined, y)
Model.score(combined, y) #array([[-3.83643468, -2.46351494]])
Model.coef_ #array([20.27202797])



''''''

import shutil

SCALE_NUM_TRIPS = 100000.0
trainsize = x
testsize = x_pred

npredictors = len(x_pred)
noutputs = 1
tf.logging.set_verbosity(tf.logging.WARN) # change to INFO to get output every 100 steps ...
shutil.rmtree('./trained_model_linear', ignore_errors=True) # so that we don't load weights from previous runs
estimator = tf.contrib.learn.LinearRegressor(model_dir='./trained_model_linear',
                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
                                             enable_centered_bias=False,
                                             feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(predictors.values))

print "starting to train ... this will take a while ... use verbosity=INFO to get more verbose output"
def input_fn(features, targets):
  return tf.constant(features.values), tf.constant(targets.values.reshape(len(targets), noutputs)/SCALE_NUM_TRIPS)
estimator.fit(input_fn=lambda: input_fn(predictors[:trainsize], targets[:trainsize]), steps=10000)

pred = np.multiply(list(estimator.predict(predictors[trainsize:].values)), SCALE_NUM_TRIPS )
rmse = np.sqrt(np.mean(np.power((targets[trainsize:].values - pred), 2)))
print 'LinearRegression has RMSE of {0}'.format(rmse)



#
SCALE_NUM_TRIPS = 100000.0
trainsize = x
testsize = x_pred
npredictors = len(x_pred)
noutputs = 1
tf.logging.set_verbosity(tf.logging.WARN) # change to INFO to get output every 100 steps ...
shutil.rmtree('./trained_model', ignore_errors=True) # so that we don't load weights from previous runs
estimator = tf.contrib.learn.DNNRegressor(model_dir='./trained_model',
                                          hidden_units=[5, 2],
                                          optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                          enable_centered_bias=False,
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(predictors.values))

print "starting to train ... this will take a while ... use verbosity=INFO to get more verbose output"
def input_fn(features, targets):
  return tf.constant(features.values), tf.constant(targets.values.reshape(len(targets), noutputs)/SCALE_NUM_TRIPS)
estimator.fit(input_fn=lambda: input_fn(predictors[:trainsize], targets[:trainsize]), steps=10000)

pred = np.multiply(list(estimator.predict(predictors[trainsize:].values)), SCALE_NUM_TRIPS )
rmse = np.sqrt(np.mean((targets[trainsize:].values - pred)**2))
print 'Neural Network Regression has RMSE of {0}'.format(rmse)

input = pd.DataFrame.from_dict(data = 
                               {'pca1' : pca1,
                                'pca2' : pca2,
                                })
# read trained model from ./trained_model
estimator = tf.contrib.learn.LinearRegressor(model_dir='./trained_model_linear',
                                          enable_centered_bias=False,
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(input.values))

pred = np.multiply(list(estimator.predict(input.values)), SCALE_NUM_TRIPS )
print pred
