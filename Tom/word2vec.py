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
from gensim.models import KeyedVectors

# Gensim
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser

# NLTK
import nltk
from nltk.corpus import stopwords
'''nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')'''
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
warnings.simplefilter("ignore", DeprecationWarning)


def clean_text(text):
    # Normalize tabs and remove newlines
    no_tabs = text.replace('\t', ' ').replace('\n', '').replace('Belt and Road', 'BRI').replace('- ', '').replace(
        'PR Newswire', '').replace('The Straits Times', '').replace('Singapore Press Holdings Limited', '').replace(
        'All Rights Reserved', '').replace('globaltimes.com.cn', '').replace('PRNewswire','');
    # Remove all characters except A-Z and a dot.
    no_url = re.sub('Online: ((www\.[^\s]+)|(https?://[^\s]+))', '', no_tabs)
    alphas_only = re.sub("[^a-zA-Z\.]", " ", no_url);
    # Normalize spaces to 1
    multi_spaces = re.sub(" +", " ", alphas_only);
    # Strip trailing and leading spaces
    no_spaces = multi_spaces.strip();

    return no_spaces


def sentence_tokenize(text):
    sentence_doc = sent_tokenize(text)
    sentences = [gensim.utils.simple_preprocess(str(doc), deacc=True) for doc in
                 sentence_doc]  # deacc=True removes punctuations
    stop = set(stopwords.words('english') + ['factiva', 'asianreview', 'viewpoint', 'sourceupdate', 'stimes', 'prn', 'st'])
    no_stop = [[word for word in sentence if word not in stop] for sentence in sentences]

    return no_stop


def lemmatization(texts, allowed_postags=['NOUN']):
    """https://spacy.io/api/annotation"""
    texts_out = [[token.lemma_ for token in text if token.pos_ in allowed_postags] for text in texts]
    return texts_out

import os

path=os.listdir(r"C:\Users\Li\Desktop\中国（txt）")
datalist=[]

for i in path:
    domain= r"C:\\Users\\Li\\Desktop\\中国（txt）\\"+i
    #print(domain)
    with open(domain,"r",encoding="utf-8") as f:
        data=f.read()
        datalist.append(data)


text_li=[clean_text(i) for i in datalist]
#print(text_li[0])
com_sent_li = [sentence_tokenize(text) for text in text_li]
#print(com_sent_li[0][0])

sent_li = []
for sentence in com_sent_li:
    for tokens in sentence:
        sent_li.append(tokens)

sent_li = [tokens for sentence in com_sent_li for tokens in sentence]
bigram = Phrases(sent_li, min_count=5, threshold=80)
trigram = Phrases(bigram[sent_li], threshold=80)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)
trigrams = [trigram_mod[bigram_mod[sentence]] for sentence in sent_li]
cores = 8


num_features = 50        # Word vector dimensionality (how many features each word will be given)
min_word_count = 2        # Minimum word count to be taken into account
num_workers = 8       # Number of threads to run in parallel (equal to your amount of cores)
context = 5              # Context window size
downsampling = 0 #1e-2    # Downsample setting for frequent words
seed_n = 1                # Seed for the random number generator (to create reproducible results)
sg_n = 1                  # Skip-gram = 1, CBOW = 0

model = Word2Vec(trigrams, workers=num_workers, min_count = min_word_count, window = context, sample = downsampling, seed=seed_n, sg=sg_n)

model.wv.save_word2vec_format('word_model.txt', binary=False)

f = open("word_model.txt", "r")
new = []
for line in f:
    new.append(line)
new[0] = '\n'
f.close()

f = open("word_model.txt", "w")
for n in new:
    f.write(n)
f.close()

import csv

with open('data.csv', 'w', newline='') as csvfile:  ##data.csv是用来存放词向量的csv文件
    writer = csv.writer(csvfile)
    data = open('word_model.txt')
    for each_line in data:
        a = each_line.split()
        writer.writerow(a)

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

l = []
words = []
with open('data.csv', 'r') as fd:
    line = fd.readline()
    line = fd.readline()
    while line:
        if line == "":
            continue
        line = line.strip()
        word = line.split(",")
        words.append(word[0])
        l.append(word[1:])
        line = fd.readline()

X = np.array(l)  # 导入数据，维度为300
pca = PCA(n_components=8)  # 降到2维
pca.fit(X)  # 训练
newX = pca.fit_transform(X)  # 降维后的数据存放在newX列表中

dict = {}
for i in range(len(words)):
    word_ = words[i]
    dict[word_] = newX[i]
for j in range(len(words)):
    print(words[j] + ':', end='')
    print(dict[words[j]])

from sklearn.cluster import KMeans
import numpy as np

X = np.array(newX)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
