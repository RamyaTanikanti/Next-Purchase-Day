#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def transform(data):
    
    # Creating new DataFrame
    user = pd.DataFrame(data['CustomerID'].unique(), columns=['CustomerID'])
    
    # Calculating Recency
    recency = data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency.columns = ['CustomerID','MaxPurchaseDate']
    recency['Recency'] = (recency['MaxPurchaseDate'].max() - recency['MaxPurchaseDate']).dt.days
    user = pd.merge(user, recency[['CustomerID','Recency']], on='CustomerID', how='left')
    
    # Calculating Frequency
    frequency = data.groupby('CustomerID')['InvoiceDate'].nunique().reset_index()
    frequency.columns = ['CustomerID', 'Frequency']
    user = pd.merge(user, frequency[['CustomerID','Frequency']], on='CustomerID', how='left')
    
    # Calculating Monetary
    data['TotalPrice'] = data['UnitPrice'] * data['Quantity']
    monetary = data.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    monetary.columns = ['CustomerID', 'Monetary']
    user = pd.merge(user, monetary[['CustomerID', 'Monetary']], on='CustomerID', how='left')
    
    # Creating New DataFrame for past Purchase Difference calculation
    days = data[['CustomerID','InvoiceDate']]
    days['InvoiceDay'] = days['InvoiceDate'].dt.date
    days.sort_values(by=['CustomerID','InvoiceDate'],inplace=True)
    
    # Dropping Duplicates as a customer must have bought different products in the same date
    days.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first',inplace=True)

    # Getting the dates of last 6 purchases
    days['PrevOrderDate'] = days.groupby('CustomerID')['InvoiceDay'].shift(1)
    days['T2OrderDate'] = days.groupby('CustomerID')['InvoiceDay'].shift(2)
    days['T3OrderDate'] = days.groupby('CustomerID')['InvoiceDay'].shift(3)
    days['T4OrderDate'] = days.groupby('CustomerID')['InvoiceDay'].shift(4)
    days['T5OrderDate'] = days.groupby('CustomerID')['InvoiceDay'].shift(5)

    # Calculating Difference between past purchase
    days['DayDiff'] = (days['InvoiceDay'] - days['PrevOrderDate']).dt.days
    days['DayDiff2'] = (days['PrevOrderDate'] - days['T2OrderDate']).dt.days
    days['DayDiff3'] = (days['T2OrderDate'] - days['T3OrderDate']).dt.days
    days['DayDiff4'] = (days['T3OrderDate'] - days['T4OrderDate']).dt.days
    days['DayDiff5'] = (days['T4OrderDate'] - days['T5OrderDate']).dt.days

    # Calculating Average and Std Deviation of all purchases for customer
    AvgStdOfPurchase = days.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
    AvgStdOfPurchase.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']
    
    # Keeping the row with each customer's latest purchase
    days.drop_duplicates(subset=['CustomerID'],keep='last',inplace=True)

    # Merging DataFrames
    days = pd.merge(days, AvgStdOfPurchase, on='CustomerID')
    user = pd.merge(user, days, on='CustomerID')
    
    # Dropping rows with missing values
    user.dropna(how='any',inplace=True)
    
    # Final DataFrame for modelling
    dfModel = user[['CustomerID','Recency','Frequency','Monetary','DayDiffMean','DayDiffStd','DayDiff5','DayDiff4','DayDiff3',
                    'DayDiff2','DayDiff']]
    return(dfModel)

