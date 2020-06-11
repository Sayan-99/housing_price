#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle


# In[28]:


warnings.filterwarnings('ignore')


# In[29]:


df = pd.read_csv('house_predict.csv')


# In[30]:


df.head()


# In[31]:


df.columns


# In[32]:


feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']


# In[33]:


X = df[feature_names]


# In[34]:


X.head()


# In[35]:


y = df['SalePrice']


# In[36]:


y


# In[37]:


X.isnull().sum()


# In[38]:


X.astype('int64')


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.75,random_state = 0)


# In[44]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 500,random_state = 1)
regressor.fit(X_train,y_train)


# In[45]:


pickle.dump(regressor,open('model.pkl','wb'))


# In[46]:


model = pickle.load(open('model.pkl','rb'))


# In[47]:


pred = regressor.predict(X_test)


# In[48]:


from sklearn.metrics import mean_absolute_error


# In[49]:


mean_absolute_error(y_test,pred)


# In[ ]:




