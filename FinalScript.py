#!/usr/bin/env python
# coding: utf-8

# In[6]:


import joblib
import pandas as pd
import datacleaning
import featureengineering
import warnings
warnings.filterwarnings('ignore')

# Reading raw data
df = pd.read_csv('TestData.csv', encoding='unicode_escape')
#print('Orginal:',df.shape)

# Cleaning
df = datacleaning.cleaning(df)

# Feature Engineering
df = featureengineering.transform(df)

# Saving CustomerID
customerID = df.pop('CustomerID')

# Loading saved model
rf = joblib.load('NextPurchaseModelRF.sav')

# Predicting on Test Data
df['Prediction'] = rf.predict(df)
df['CustomerID'] = customerID
df['RFMCluster'] = rfmCluster

# Saving Predictions as csv file
df.to_csv('NextPurchasePredictions.csv',index=False)