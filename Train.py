import pandas as pd
import numpy as np
import datacleaning
import featureengineering
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('TrainData.csv')
# Cleaning Data using the Cleaning module
train_cleaned = datacleaning.cleaning(train)
# Creating new features using Feature Engineering module
train_processed = featureengineering.transform(train_cleaned)

train_processed.drop('CustomerID', axis=1,inplace=True)
X = train_processed.drop('DayDiff', axis=1)
y = train_processed['DayDiff']

rf = RandomForestClassifier(random_state=11, max_depth=17, min_samples_leaf=4, min_samples_split=5, n_estimators=165)
rf.fit(X,y)
dump(rf, 'NextPurchaseModelRF.sav') 