#!/usr/bin/env python
# coding: utf-8

# In[6]:
#ln5

import pandas as pd


# In[7]:


import numpy as np


# In[13]:


import seaborn as sns


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv('KNN_Project_Data')


# In[14]:


df.head()


# In[15]:


sns.pairplot(df,hue='TARGET CLASS')


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()


# In[18]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[21]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[26]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head(3)


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X  = df_feat
y = df['TARGET CLASS']


# In[29]:


X_train ,X_test, y_train ,y_test = train_test_split(X,y,test_size=30,random_state=101)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[31]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[35]:


knn.fit(X_train,y_train)


# In[36]:


pred = knn.predict(X_test)


# In[38]:


from sklearn.metrics import classification_report,confusion_matrix


# In[39]:


print(confusion_matrix(y_test,pred))


# In[40]:


print(classification_report(y_test,pred))


# In[42]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i!= y_test)) 


# In[55]:


plt.figure(figsize=(10,20))
plt.plot(range(1,40),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='yellow',markersize=10)
plt.title('Error Rate vs K')
plt.xlabel('k')


# In[56]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




