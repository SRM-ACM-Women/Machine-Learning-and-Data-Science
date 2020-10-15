#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('stopwords')


# In[2]:


#python3 -m spacy download en


# In[3]:


import re
import pandas as pd
import numpy as np
import gensim.corpora as corpora
import gensim.models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[4]:


import spacy


# In[5]:


import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[7]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[8]:


df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())
df.head()


# In[9]:


data = df.content.values.tolist()


# In[10]:


data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]


# In[11]:


data = [re.sub("\'", "", sent) for sent in data]


# In[12]:


data = [re.sub('\s+', ' ', sent) for sent in data]


# In[13]:


print(data[:2])


# In[14]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence),deacc=True))


# In[15]:


data_words=list(sent_to_words(data))
print(data_words[:1])


# In[ ]:





# In[16]:


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  


# In[17]:


bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[18]:


print(trigram_mod[bigram_mod[data_words[0]]])


# In[28]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[29]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# In[19]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# In[20]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


# In[21]:


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[31]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[23]:


data_words_nostops = remove_stopwords(data_words)


# In[24]:


data_words_bigrams = make_bigrams(data_words_nostops)
print(data_words_bigrams[:1])


# In[25]:


nlp=spacy.load('en',disable=['parser','ner'])


# In[ ]:





# In[26]:


data_lemmmatized=lemmatization(data_words_bigrams,allowed_postags=['NOUN','ADJ','VERB','ADJ'])


# In[27]:


print(data_lemmmatized[:1])


# In[29]:


id2word=corpora.Dictionary(data_lemmmatized)    


# In[30]:


texts=data_lemmmatized


# In[31]:


corpus=[id2word.doc2bow(text) for text in texts]


# In[32]:


print(corpus[:1])


# In[33]:


id2word[0]


# In[34]:


for cp in corpus[:1]:
    for id,freq in cp:
        print(id2word[id], freq)



#[[print(id2word[id] for id, freq in cp for cp in corpus[:1]]


# In[35]:


lda_model=gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id2word,
                                          num_topics=20,
                                         random_state=100,
                                         update_every=1,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         per_word_topics=True)


# In[45]:


print(lda_model.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False))
doc_lda=lda_model[corpus]


# In[41]:


for i in lda_model.print_topics():
          print(i)
          print('\n')


# In[32]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:
