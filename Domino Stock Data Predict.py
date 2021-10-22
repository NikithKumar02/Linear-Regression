#!/usr/bin/env python
# coding: utf-8

# # VAC ASSIGNMENT
# # Domino Stock Data

# In[7]:


#import python libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


dataset = pd.read_csv('Dominos_Stock_Data.csv')


# In[10]:


dataset.head()


# In[12]:


dataset['Date'] = pd.to_datetime(dataset.Date)


# In[13]:


dataset.shape


# In[14]:


dataset.drop('Adj Close', axis = 1, inplace = True)


# In[15]:


dataset.head()


# In[16]:


dataset.isnull().sum()


# In[17]:


dataset.isna().any()


# In[18]:


dataset.info()


# In[19]:


dataset.describe()


# In[20]:


dataset['Open'].plot(figsize=(18, 7))


# In[22]:


x = dataset[['Open', 'High', 'Low', 'Volume']]
y = dataset['Close']


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[34]:


x_train.shape


# In[36]:


x_test.shape


# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regression = LinearRegression()


# In[40]:


regression.fit(x_train, y_train)


# In[41]:


print(regression.coef_)


# In[42]:


print(regression.intercept_)


# In[52]:


predict = regression.predict(x_test)


# In[54]:


print(x_test)


# In[47]:


predict.shape


# In[55]:


dframe = pd.DataFrame(y_test, predict)


# In[57]:


dfr = pd.DataFrame({'Actual Price':y_test, 'Predicted Price':predict})


# In[58]:


print(dfr)


# In[59]:


dfr.head(30)


# In[60]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[61]:


regression.score(x_test, y_test)


# In[62]:


import math


# In[63]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predict))


# In[64]:


print('Mean Squared Error:', metrics.mean_squared_error(y_test, predict))


# In[65]:


print('Root Mean Absolute Error:', math.sqrt(metrics.mean_absolute_error(y_test, predict)))


# In[66]:


graph = dfr.head(25)


# In[68]:


graph.plot(kind='bar', figsize=(16, 7))


# In[ ]:




