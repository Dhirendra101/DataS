
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import nltk
spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
spam_data.describe()


# In[28]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# In[29]:


def answer_one():
    
    total=spam_data.shape[0]
    #cnt_spam = len(spam_data[spam_data['target'] == 1])
    cnt_spam =spam_data[spam_data['target'] == 1].count()
    spam_per=(cnt_spam*100)/total
    #print(total_spam)
    return total,cnt_spam,spam_per
    


# In[30]:


answer_one()


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vect = CountVectorizer().fit(X_train)
    words=vect.get_feature_names()
    token=len(words)
    word_dist = nltk.FreqDist(words)
    rslt=pd.DataFrame(word_dist.most_common(2))
    print(rslt)
    length=len(vect.get_feature_names())
    #print(rslt)
    return length,vect.get_feature_names(),word_dist,rslt,token


# In[6]:


length,name,word_dist,rslt,token=answer_two()
length,token


# In[35]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    x_train_vect =vect.transform(X_train)
    x_test_vect = vect.transform(X_test)
    clfNaive=MultinomialNB(alpha=0.1)
    clfNaive.fit(x_train_vect,y_train)
    predictions=clfNaive.predict(x_test_vect)
    roc= roc_auc_score(y_test, predictions)
    return roc,x_train_vect.shape,predictions,x_train_vect


# In[36]:


a,c,predictions,x_train_vect=answer_three()
a
#pd.DataFrame(X_test,predictions)


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    tfvect=TfidfVectorizer().fit(X_train)
    x_train_tfvect=tfvect.transform(X_train)
    x_test_tfvect=tfvect.transform(X_test)
    name=tfvect.get_feature_names()
    feature_names = np.array(tfvect.get_feature_names())
    sorted_tfidf_index = x_train_tfvect.max(0).toarray()[0].argsort()
    
    return feature_names,sorted_tfidf_index


# In[10]:


feature_names,sorted_tfidf_index=answer_four()
print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


# In[11]:


print('3')


# In[12]:


def answer_five():
    tfvect=TfidfVectorizer(min_df=5).fit(X_train)
    x_train_tfvect=tfvect.transform(X_train)
    x_test_tfvect=tfvect.transform(X_test)
    clfNaive=MultinomialNB(alpha=0.1)
    clfNaive.fit(x_train_tfvect,y_train)
    predictions=clfNaive.predict(x_test_tfvect)
    roc= roc_auc_score(y_test, predictions)
    
    
    
    return roc


# In[13]:


answer_five()


# In[14]:


def answer_six():
    cnt_spam =spam_data[spam_data['target'] == 1].count()
    spam_data['char_count'] = spam_data['text'].str.len()
    
    return cnt_spam


# In[15]:


answer_six()
print(spam_data.head())
spam_data.groupby(['target'])['char_count'].mean()
#spam_data['char_count'].mean()

