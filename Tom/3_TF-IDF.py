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


def clean_text(text):
    # Normalize tabs and remove newlines
    no_tabs = text.replace('\t', ' ').replace('\n', ' ').replace('- ', '')
    # Remove all characters except A-Z and a dot.
    no_url = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', no_tabs)
    alphas_only = re.sub("[^a-zA-Z]", " ", no_url);
    # Normalize spaces to 1
    multi_spaces = re.sub(" +", " ", alphas_only);
    # Strip trailing and leading spaces
    no_spaces = multi_spaces.strip();
    return no_spaces


with open(r"C:\Users\Li\Desktop\中国（txt）\all.txt", "r", encoding="utf-8") as f:
    china_corpus = f.read()

with open(r"C:\Users\Li\Desktop\美国（txt）\all.txt", "r", encoding="utf-8") as f:
    us_corpus = f.read()

with open(r"C:\Users\Li\Desktop\新加坡（txt）\all.txt", "r", encoding="utf-8") as f:
    singa_corpus = f.read()


china_cor=clean_text(china_corpus)
us_cor=clean_text(us_corpus)
singa_cor=clean_text(singa_corpus)

test_list = [china_cor, us_corpus, singa_corpus]

#print(us_cor)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from  sklearn.feature_extraction.text import TfidfVectorizer
# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(decode_error='ignore', stop_words='english',smooth_idf=True,use_idf=True)

# fit and transform the texts
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(test_list)
# get the vector for the third document
vector_tfidfvectorizer = tfidf_vectorizer_vectors[0] # Note that 2 refers to document3, due to zero-based indexing

# place tf-idf values in a DataFrame
df = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
C=df.sort_values(by=["tfidf"],ascending=False)[:50]
print('中国',C)
C.to_csv(r"C:\Users\Li\Desktop\新加坡（txt）\china.csv")

# get the vector for the third document
vector_tfidfvectorizer = tfidf_vectorizer_vectors[1] # Note that 2 refers to document3, due to zero-based indexing

# place tf-idf values in a DataFrame
df = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
U=df.sort_values(by=["tfidf"],ascending=False)[:50]
print('美国',U)
U.to_csv(r"C:\Users\Li\Desktop\新加坡（txt）\singa.csv")

# get the vector for the third document
vector_tfidfvectorizer = tfidf_vectorizer_vectors[2] # Note that 2 refers to document3, due to zero-based indexing

# place tf-idf values in a DataFrame
df = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
S=df.sort_values(by=["tfidf"],ascending=False)[:50]
print('美国',S)
S.to_csv(r"C:\Users\Li\Desktop\新加坡（txt）\us.csv")