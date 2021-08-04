import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import pandas as pd
import os
import pickle
import re
import string
import time

# Gensim
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser


from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import os


def clean_text(text):
    # Normalize tabs and remove newlines
    no_tabs = text.replace('\t', ' ').replace('\n', ' ').replace('Belt and Road', 'BRI').replace('- ', '').replace('globaltimes.com.cn', ' ').replace('SOURCE BUSINESS SPOTLIGHT',' ')
    # Remove all characters except A-Z and a dot.
    no_url = re.sub('Online: ((www\.[^\s]+)|(https?://[^\s]+))', '', no_tabs)
    alphas_only = re.sub("[^a-zA-Z\.]", " ", no_url);
    # Normalize spaces to 1
    multi_spaces = re.sub(" +", " ", alphas_only);
    # Strip trailing and leading spaces
    no_spaces = multi_spaces.strip();

    return no_spaces


path=os.listdir(r"C:\Users\Li\Desktop\中国（txt）")
datalist=[]

for i in path:
    domain= r"C:\\Users\\Li\\Desktop\\中国（txt）\\"+i
    #print(domain)
    with open(domain,"r",encoding="utf-8") as f:
        data=f.read()
        datalist.append(data)


def token_text(text):
    word=text.lower
    word_doc = word_tokenize(word)
    stop = set(stopwords.words('english') + ['factiva', 'asianreview', 'viewpoint', 'sourceupdate', 'stimes', 'prn', 'gt','page','editor'])
    no_stop = [i for i in word_doc if i not in stop]
    return no_stop

