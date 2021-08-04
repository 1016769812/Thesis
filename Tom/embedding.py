# General
from pprint import pprint
from collections import Counter
import os
import re
import logging
import string
import pickle
import numpy as np
import pandas as pd
import smart_open
import multiprocessing
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

# Gensim
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser

import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Spacy
import spacy

# Plotting
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value

# Clustering
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE

# Suppressing warnings
import warnings

df_com = pd.read_csv(r"C:\Users\Li\Desktop\icc3-comments.csv", lineterminator='\n')
df_com = df_com[~df_com['text'].isin(['[removed]', '[deleted]' ])].dropna(subset=['text']).reset_index(drop=True)

def clean_text(text):
    # Normalize tabs and remove newlines
    no_tabs = text.replace('\t', ' ').replace('\n', '');
    # Remove all characters except A-Z and a dot.
    alphas_only = re.sub("[^a-zA-Z\.]", " ", no_tabs);
    # Normalize spaces to 1
    multi_spaces = re.sub(" +", " ", alphas_only);
    # Strip trailing and leading spaces
    no_spaces = multi_spaces.strip();
    return no_spaces

df_com["text_clean"] = df_com["text"].apply(lambda x: clean_text(x))
text_li = df_com['text_clean'].tolist()

def sentence_tokenize(text):
    sentence_doc = sent_tokenize(text)
    sentences = [gensim.utils.simple_preprocess(str(doc), deacc=True) for doc in sentence_doc]  # deacc=True removes punctuations
    stop = set(stopwords.words('english') + ['’', '“', '”', 'nbsp', 'http'])
    no_stop = [[word for word in sentence if word not in stop] for sentence in sentences]
    return no_stop

com_sent_li = [sentence_tokenize(text) for text in text_li]
#print(com_sent_li)

sent_li = []
for sentence in com_sent_li:
    for tokens in sentence:
        sent_li.append(tokens)
sent_li = [tokens for sentence in com_sent_li for tokens in sentence]
print(sent_li)

bigram = Phrases(sent_li, min_count=5, threshold=80)
trigram = Phrases(bigram[sent_li], threshold=80)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)
trigrams = [trigram_mod[bigram_mod[sentence]] for sentence in sent_li]
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
num_features = 500        # Word vector dimensionality (how many features each word will be given)
min_word_count = 2        # Minimum word count to be taken into account
num_workers = cores       # Number of threads to run in parallel (equal to your amount of cores)
context = 10              # Context window size
downsampling = 0 #1e-2    # Downsample setting for frequent words
seed_n = 1                # Seed for the random number generator (to create reproducible results)
sg_n = 1                  # Skip-gram = 1, CBOW = 0

model = Word2Vec(trigrams, workers=num_workers,
            size=num_features, min_count = min_word_count,
            window = context, sample = downsampling, seed=seed_n, sg=sg_n)

model.save("word2vec.vec")
model = Word2Vec.load("word2vec.vec")

def get_related_terms(token, topn=20):
    """
    look up the topn most similar terms to token and print them as a formatted list
    """

    for word, similarity in model.most_similar(positive=[token], topn=topn):
        print(word, round(similarity, 3))

get_related_terms(u'culture')


def word_algebra(add=[], subtract=[], topn=1):
    """
    combine the vectors associated with the words provided
    in add= and subtract=, look up the topn most similar
    terms to the combined vector, and print the result(s)
    """
    answers = model.most_similar(positive=add, negative=subtract, topn=topn)

    for term, similarity in answers:
        print(term)


word_algebra(add=['cultural','problem'])
word_algebra(add=['cultural','problem'], subtract=['man'])