#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def cleaning(data):
    
    # Dropping rows where CustomerID is missing
    data.drop(data[data['CustomerID'].isnull()].index,inplace=True)
    
    # Changing data-type for Date Column
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    
    # Cleaning Description column
    data['Description'] = data['Description'].str.upper()
    data['Description'] = data['Description'].str.strip()
    
    # Cleaning StockCode column
    data['StockCode'] = data['StockCode'].str.upper()
    data.drop(data[data['StockCode']=='D'].index, inplace=True)
    data.drop(data[data['StockCode']=='POST'].index, inplace=True)
    data.drop(data[data['StockCode']=='DOT'].index, inplace=True)
    data.drop(data[data['StockCode']=='CRUK'].index, inplace=True)
    data.drop(data[data['StockCode']=='DOTCOM POSTAGE'].index, inplace=True)
    data.drop(data[data['StockCode']=='BANK CHARGES'].index, inplace=True)
    
    # Cleaning Quantity and UnitPrice column
    data.drop(data[data['Quantity']<=0].index,inplace=True)
    data.drop(data[data['UnitPrice']<=0].index,inplace=True)
    
    # Creating new Country column
    data['CountryNew'] = np.where(data['Country']!='United Kingdom', 'Others', data['Country'])
    
    return(data)

