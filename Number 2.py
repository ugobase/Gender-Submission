#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Read CSV file
Gen = pd.read_csv('gender_submission.csv')


# In[3]:


#First five rows of csv file
Gen.head()


# In[4]:


#Last five rows of csv file
Gen.tail()


# In[5]:


#Replace PassengerId with Passenger_Id
Gen.columns = Gen.columns.str.replace('PassengerId', 'Passenger_Id')
Gen


# In[6]:


#Number of rows
Gen.shape[0]


# In[7]:


#Number of columns
Gen.shape[1]


# In[8]:


#General information of the file
Gen.info()


# In[9]:


#Statistical description
Gen.describe()


# In[10]:


#Top 10 largest values of Survived
Gen.nlargest(n=10, columns=['Survived'])


# In[11]:


#Alternative to previous
Gen.sort_values('Survived', ascending = False).head(10)


# In[12]:


#Top 5 smallest values of Passenger_Id
Gen.nsmallest(5,['Passenger_Id'])


# In[13]:


#Alternative to previous
Gen.sort_values('Passenger_Id', ascending = True).head(5)


# In[14]:


#Number of Survived = 1
Gen.loc[(Gen['Survived'] == 1)].shape[0]


# In[15]:


#Alternative to previous
Gen[Gen.Survived == 1].shape[0]


# In[16]:


#Number of Survived = 0
Gen.loc[(Gen['Survived'] == 0)].shape[0]


# In[17]:


#Alternative to previous
Gen[Gen.Survived == 0].shape[0]


# In[18]:


#Using map to give variables for survived
Gen['Survived'].map({0: 'Died', 1: 'Lived'})


# In[19]:


#Alternative to previous
Gen.Survived.map({0: 'Died', 1: 'Lived'})


# In[20]:


#Number of Passenger_Id > 1000
Gen.loc[(Gen['Passenger_Id'] > 1000)].shape[0]


# In[21]:


#Alternative to previous
Gen[Gen.Passenger_Id > 1000].shape[0]


# In[22]:


#Number of Passenger_Id >= 1000 and show Passenger_Id column only
Gen.loc[(Gen['Passenger_Id'] >= 1000), 'Passenger_Id']


# In[23]:


#Alternative to previous
Gen[Gen.Passenger_Id >= 1000].Passenger_Id


# In[24]:


#Pasenger_Id >= 1300
Gen.loc[(Gen['Passenger_Id'] >= 1300)]


# In[25]:


#Alternative to previous
Gen[Gen.Passenger_Id >= 1300].Passenger_Id


# In[26]:


#Number of Passenger_Id > 1200 and survived = 1
Gen.loc[(Gen['Passenger_Id'] > 1200) & (Gen['Survived'] == 1)].shape[0]


# In[27]:


#Alternative to previous
Gen[(Gen.Passenger_Id > 1200) & (Gen.Survived == 1)].shape[0]


# In[28]:


#Number of Passenger_Id > 1200 or survived = 1
Gen.loc[(Gen['Passenger_Id'] > 1200) | (Gen['Survived'] == 1)].shape[0]


# In[29]:


#Alternative to previous
Gen[(Gen.Passenger_Id > 1200) | (Gen.Survived == 1)].shape[0]


# In[30]:


#Grouping and counting survived column by Passenger_Id > 1200
Gen.loc[(Gen['Passenger_Id'] > 1200)].groupby(['Survived']).count()


# In[31]:


#Alternative to previous
Gen[Gen.Passenger_Id > 1200].Survived.value_counts()


# In[32]:


#If condition where 1 = Alive and 0 = Dead using a set of numbers to create a new column
Gen.loc[(Gen['Survived'] == 1), 'New_Col'] = 'Alive'
Gen.loc[(Gen['Survived'] == 0), 'New_Col'] = 'Dead'
Gen


# In[33]:


#Alternative to previous using insert and lambda expression
Gen.insert(2, 'New', Gen.Survived.apply(lambda x: 'Alive' if x == 1 else 'Dead'))
Gen


# In[34]:


#if condition when Passenger_Id > 1200 True and < 1200 is false using lambda expressions
Gen['Next_One'] = Gen['Passenger_Id'].apply(lambda x: 'True' if x > 1200 else 'False')
Gen


# In[35]:


#Alternative to previous
Gen.insert(5, 'Next', Gen.Passenger_Id.apply(lambda x: 'True' if x > 1200 else 'False'))
Gen


# In[36]:

#Histogram plot of survived
ax = Gen['Survived'].plot(kind = 'hist', figsize = (8,6))
ax.set_ylabel ('Next')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[37]:

#Alternative to previous using seaborn
plt.figure(figsize=(12,8))
sns.histplot(Gen.Survived)


# In[38]:

#Density plot of survived
ax = Gen['Survived'].plot(kind = 'density', figsize = (8,6))
ax.set_ylabel ('Next')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[39]:

#Alternative to previous using seaborn
plt.figure(figsize=(12,8))
sns.displot(Gen.Survived, kind = 'kde')


# In[40]:

#Box plot of passenger_Id
ax = Gen['Passenger_Id'].plot(kind = 'box', figsize = (8,6))
ax.set_ylabel ('Next')
plt.title('Don', loc = 'right')


# In[41]:

#Box plot of survived against passenger_Id
plt.figure(figsize=(12,8))
sns.boxplot(data = Gen, x = 'Survived', y = 'Passenger_Id')


# In[42]:

#Bar chart of survived
Gen['Survived'].value_counts().plot(kind = 'bar', figsize = (6,6))


# In[43]:

#Pie chart of survived
Gen['Survived'].value_counts().plot(kind = 'pie', figsize = (6,6))


# In[44]:


#Counting all values of survived column and grouping them into 0 and 1
Gen['Survived'].value_counts()


# In[45]:


#Unique values of survived column
Gen['Survived'].unique()


# In[46]:

#Correlation of the dataset
plt.figure(figsize = (8,6))
sns.heatmap(Gen.corr(), annot = True, fmt = '0.1f')


# In[47]:

#Count plot of survived
sns.countplot(Gen['Survived'])


# In[ ]:




