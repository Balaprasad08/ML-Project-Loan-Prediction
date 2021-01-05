#!/usr/bin/env python
# coding: utf-8

# #### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os


# In[2]:


os.chdir('E:\\prasad\\practice\\My Working Projects\\Completed\\Loan Prediction')


# In[3]:


df=pd.read_csv('Loan Prediction Dataset.csv')


# In[4]:


df.shape


# In[5]:


df.head(2)


# In[6]:


df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())


# In[7]:


df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median())


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# In[11]:


df.drop('Loan_ID',inplace=True,axis=1)


# In[12]:


df.head(2)


# In[13]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler


# In[14]:


le=LabelEncoder()


# In[15]:


df['Gender']=le.fit_transform(df['Gender'])


# In[16]:


df['Married']=le.fit_transform(df['Married'])


# In[17]:


df['Education']=le.fit_transform(df['Education'])


# In[18]:


df['Self_Employed']=le.fit_transform(df['Self_Employed'])


# In[19]:


df['Property_Area']=le.fit_transform(df['Property_Area'])


# In[20]:


df['Loan_Status']=le.fit_transform(df['Loan_Status'])


# In[21]:


df.head(2)


# In[22]:


X=df.iloc[:,:-1]
X.head(2)


# In[23]:


X['Dependents'].unique()


# In[24]:


X['Dependents']=X['Dependents'].map({'0':0, '1':1, '2':2, '3+':3})


# In[25]:


y=df.iloc[:,-1]
y.head(2)


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[28]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[29]:


sc=StandardScaler()


# In[30]:


X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.fit_transform(X_test)


# In[31]:


X_train=pd.DataFrame(X_train_sc,columns=X_train.columns)
X_test=pd.DataFrame(X_test_sc,columns=X_test.columns)


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[33]:


X_train.head(2)


# In[40]:


def check_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('accuracy_score:',accuracy_score(y_test,y_pred))


# In[41]:


check_model(RandomForestClassifier(),X_train,y_train)


# In[42]:


check_model(LogisticRegression(),X_train,y_train)


# In[43]:


check_model(SVC(),X_train,y_train)


# In[44]:


check_model(DecisionTreeClassifier(),X_train,y_train)


# ### Final Model

# In[45]:


model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('accuracy_score:',accuracy_score(y_test,y_pred))


# ### Save Model Pickle & Joblib

# In[46]:


import pickle,joblib


# In[47]:


pickle.dump(model,open('loan_pred.pickle','wb'))


# In[48]:


joblib.dump(model,'loan_jb.joblib')


# #### Load Pickle Model

# In[49]:


model_pkl=pickle.load(open('loan_pred.pickle','rb'))


# In[50]:


y_pred_pickle=model_pkl.predict(X_test)


# In[51]:


accuracy_score(y_test,y_pred_pickle)


# #### Load Joblib Model

# In[52]:


model_jbl=joblib.load('loan_jb.joblib')


# In[53]:


y_pred_jbl=model_jbl.predict(X_test)


# In[54]:


accuracy_score(y_test,y_pred_jbl)


# In[ ]:




