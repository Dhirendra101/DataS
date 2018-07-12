
# coding: utf-8

# In[307]:


import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from nltk import sent_tokenize
from nltk import word_tokenize

def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
   
   
    synonyms = []
    doc_comp=word_tokenize(doc)
    poss=nltk.pos_tag(doc_comp)
    #print(poss)
    for doc in poss:
        #print(doc)
        #poss=nltk.pos_tag(doc)
        #print(poss)
        wordnet_tag=convert_tag(doc[1][0])
        #print(wordnet_tag)
        #print(poss)
        synon=wn.synsets(doc[0],wordnet_tag)
        #print(synon)
        if len(synon)!=0:
            synonyms.append(synon[0])
            
        
        
        #for syn in wn.synsets(doc[0],wordnet_tag):
            #for i in syn.lemmas():
                #synonyms.append(i.name()) 
                #print(synonyms)
            
         
            
                

    # Your Code Here
    #print(synonyms)
    return synonyms# Your Answer Here


def similarity_score(s1, s2):
    
    get_score=[]
    max_score=[]
    for syn1 in s1:
        for syn2 in s2:
            #print(syn1,syn2)
            score=wn.path_similarity(syn1,syn2)
            #print('score is',score)
            if score is not None:
                #print('true score',score)
                get_score.append(score)
                #print(score)
                #print('hi')
            #print(get_score.append(score))
            
        #print(get_score)    
        if len(get_score)>=1:
            #print('hi')
            max_score.append(max(get_score))
            
        
           
    
    #print(nltk.pos_tag(s1))
    # Your Code Here
    
    return (sum(max_score)/len(max_score))    # Your Answer Here


def document_path_similarity(doc1, doc2):
   

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    print(similarity_score(synsets1, synsets2))
    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


# In[308]:


dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
wn.path_similarity(dog,cat)
#doc='function'
#tok=word_tokenize(doc)
#poss=nltk.pos_tag(tok)
#print(poss[0][1])
#word=poss[0]
#print(type(word))
#wn.synsets(word+'.n')
#poss
get_score=[]
a=wn.synset('function.n.01')
b=wn.synset('see.v.01')
r=wn.path_similarity(a,b)
print(r)
if r is not None:
    print('hi')
    
#print(get_score)    
#print(wn.synsets('dog','n')[0])
#synonyms=[]
#for syn in wn.synsets('dog','n'):
    #for i in syn.lemmas():
        #synonyms.append(i.name())
            
#synonyms[0]                


# In[309]:


def test_document_path_similarity():
    #doc1='function'
    #doc2='see'
    doc1 = 'what is your name'
    doc2 = 'May I know your name'
    return document_path_similarity(doc1, doc2)


# In[310]:


result=test_document_path_similarity()
print('result is:',result)


# In[334]:


# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.info()


# In[337]:


def most_similar_docs():
    result_list=[]
    que1=paraphrases['D1'].tolist()
    que2=paraphrases['D2'].tolist()
    #result=document_path_similarity(que1[1],que2[2])
    for i in range(len(que1)):
        result=document_path_similarity(que1[i],que2[i])
        result_list.append(result)
         
    #result=document_path_similarity(paraphrases['D1'].astype(str), paraphrases['D2'].astype(str))
    #print(que1)
    
    # Your Code Here
    
    return result_list


# In[1]:


result=most_similar_docs()
paraphrases['result']=result
paraphrases


# In[ ]:


def label_accuracy():
    from sklearn.metrics import accuracy_score

    # Your Code Here
    
    return # Your Answer Here


# In[354]:


import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english')

#token_pattern='(?u)\\b\\w\\w\\w+\\b'
#print(newsgroup_data)
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())
#print(id_map)


# In[359]:


# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
#ldamodel = 
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(corpus, num_topics=30, id2word = id_map, passes=25)
ldamodel


# In[360]:


def lda_topics():
    top=ldamodel.print_topics(4)
    
    # Your Code Here
    
    return top# Your Answer Here


# In[363]:


arr=lda_topics()
arr[1]


# In[366]:


new_doc = ["\n\nIt's my understanding that the freezing will start to occur because of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge Krumins\n-- "]


# In[391]:


def topic_distribution():
    vect_new = CountVectorizer(stop_words='english')
    X_new = vect_new.fit_transform(new_doc)

# Convert sparse matrix to gensim corpus.
    corpus_new = gensim.matutils.Sparse2Corpus(X_new, documents_columns=False)
    
    #ldamodel(corpus_new, num_topics=5)
    #arr1=ldamodel.print_topics(4)(corpus_new)
    arr1=ldamodel.get_document_topics(corpus_new)
    
    # Your Code Here
    
    return arr1 # Your Answer Here


# In[394]:


arr2=topic_distribution()
for ar in range(len(arr2)):
    print(arr2[ar])


# In[ ]:


def topic_names():
    
    # Your Code Here
    
    return # Your Answer Here

