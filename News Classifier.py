#!/usr/bin/env python
# coding: utf-8

# Importing pandas to convert csv file to dataframe

# In[2]:


import pandas as pd
import numpy as np


# Importing news.csv and dropping unnamed index column

# In[40]:


df_news=pd.read_csv('news.csv')
df_news=df_news.drop(['Unnamed: 0'],axis=1)
df_news


# Importing another dataset for fake news

# In[13]:


df_fake=pd.read_csv('fake.csv')
df_fake


# Dropping subject and date columns from fake.csv and adding label as FAKE

# In[14]:


df_fake=df_fake.drop(['subject','date'],axis=1)


# In[41]:


df_fake['label']='FAKE'
df_fake


# Repeating the same process as above for real.csv dataset

# In[16]:


df_real=pd.read_csv('true.csv')
df_real=df_real.drop(['subject','date'],axis=1)
df_real['label']='REAL'
df_real


# In[17]:


df_news


# Merging the three dataframes to form one large news.csv dataframe

# In[18]:


df_news=pd.concat([df_news, df_fake, df_real])
df_news


# Importing sklearn libraries

# In[36]:


import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools


# In[24]:


label=df_news.label
label.head()


# Seperating labels and initialising up training and testing datasets

# In[25]:


x_train,x_test,y_train,y_test=train_test_split(df_news['text'], label, test_size=0.2, random_state=7)


# Initialising a TF-IDF vectorizer with a maximum document frequency of 0.7 (70%)

# In[26]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[27]:


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[33]:


print(tfidf_test)


# Initialising a Passive Aggressive classifier

# In[29]:


psg=PassiveAggressiveClassifier(max_iter=50)
psg.fit(tfidf_train,y_train)


# In[32]:


y_pred=psg.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print("Accuracy: {}%".format(round(score*100,2)))


# An accuracy of 97.6% is obtained from the augmented dataset

# Setting up a confusion matrix to evaluate performance of the classifier

# In[34]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[37]:


pred = psg.predict(tfidf_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[ ]:




