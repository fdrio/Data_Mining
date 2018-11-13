#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division
from statistics import mode
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform
import numbers


# ### Reading data set

# In[3]:


path = os.getcwd()
datasetpath = os.path.join(path,"data")
datasetpath = os.path.join(datasetpath,"2006.csv")
flights = pd.read_csv(datasetpath)
flights


# In[3]:


l = flights.columns[flights.isnull().any()].tolist()
l


# In[4]:


from collections import Counter
for col in l[:-1]:
    print(flights[col].isna().sum())


# In[1]:


flights.iloc[1]


# In[4]:


# dropping column
flightdf = flights.drop("CancellationCode",axis=1)
flightdf = flightdf.dropna()


# In[7]:


flightdf


# In[ ]:





# In[5]:


from sklearn.preprocessing import LabelEncoder
ce = LabelEncoder()

flightdf["UniqueCarrier"] = ce.fit_transform(flightdf["UniqueCarrier"])
flightdf["TailNum"] = ce.fit_transform(flightdf["TailNum"])
flightdf["Origin"] = ce.fit_transform(flightdf["Origin"])
flightdf["Dest"] = ce.fit_transform(flightdf["Dest"])


# In[9]:


flightdf.dtypes


# In[6]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(flightdf,test_size=0.25)
train_labels = train["ActualElapsedTime"]
test_labels = test["ActualElapsedTime"]


# In[11]:


# Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train.drop("ActualElapsedTime",axis=1),train_labels)


# In[12]:


gnb_pred = gnb.predict(test.drop("ActualElapsedTime",axis=1))


# In[13]:


(gnb_pred == test_labels).sum()/len(test_labels)


# In[ ]:


from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(train.drop("ActualElapsedTime",axis=1),train_labels)


# In[15]:


dt_pred = dt.predict(test.drop("ActualElapsedTime",axis=1))


# In[16]:


(dt_pred == test_labels).sum()/len(test_labels)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(train.drop("ActualElapsedTime",axis=1),train_labels)


# In[18]:


knnpred = clf.predict(test.drop("ActualElapsedTime",axis=1)[:500000])
(knnpred == test_labels[:500000]).sum()/len(test_labels[:500000])


# In[7]:


# with feature selection
from sklearn.ensemble import ExtraTreesClassifier
clf=ExtraTreesClassifier(n_estimators=10) # selecting the 10 best features
clf.fit(flightdf.drop("ActualElapsedTime",axis=1)[:500000],flightdf["ActualElapsedTime"][:500000])


# In[8]:


clf.feature_importances_# feature importance


# In[9]:


from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf,prefit=True)
new=model.transform(flightdf.drop("ActualElapsedTime",axis=1))
new.shape
labels = flightdf["ActualElapsedTime"]
frames = [pd.DataFrame(new),labels]
new_df = pd.concat(frames,axis=1)
new_df




from sklearn.model_selection import train_test_split
train,test = train_test_split(flightdf,test_size=0.25)
train_labels = train["ActualElapsedTime"]
test_labels = test["ActualElapsedTime"]


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


dt = tree.DecisionTreeClassifier()
dt.fit(train.drop("ActualElapsedTime",axis=1),train_labels)
dt_pred = dt.predict(test.drop("ActualElapsedTime",axis=1))
acc = (dt_pred == test_labels).sum()/len(test_labels)
print(acc)


# In[ ]:


gnb = GaussianNB()
gnb.fit(train.drop("ActualElapsedTime",axis=1),train_labels)
gnb_pred = gnb.predict(test.drop("ActualElapsedTime",axis=1))
acc = (gnb_pred == test_labels).sum()/len(test_labels)
print(acc)


# In[ ]:


clf = KNeighborsClassifier()
clf.fit(train.drop("ActualElapsedTime",axis=1),train_labels)
knnpred = clf.predict(test.drop("ActualElapsedTime",axis=1))
acc =(knnpred == test_labels).sum()/len(test_labels)
print(acc)

