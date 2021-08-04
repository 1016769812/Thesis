import nltk

'''nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')'''

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

trp = pd.read_csv(r"C:\Users\Li\Desktop\Tom\seduction.csv", lineterminator="\n")
sed = pd.read_csv(r"C:\Users\Li\Desktop\Tom\seduction.csv", lineterminator="\n")
mgtow = pd.read_csv(r"C:\Users\Li\Desktop\Tom\mgtow.csv", lineterminator="\n")

data = {'Name':['Sai', 'Jack', 'Angela', 'Matt', 'Alisha', 'Ricky'],'Age':[28,34,None,42, "[removed]", "[deleted]"]}
df = pd.DataFrame(data)
print(df)
clean_df = df.dropna(subset=['Age'])

cleaner_df = clean_df[~clean_df['Age'].isin(['[removed]', '[deleted]' ])]
print(cleaner_df)
print("seduction: " + str(len(sed)))
print("theredpill: " + str(len(trp)))
print("mgtow: " + str(len(mgtow)))

def preprocessing(df):
    """POS tags and filters DF by nouns"""
    dfLength = len(df)
    total = ""
    counter = 0
    clean = df[~df['selftext'].isin(['[removed]', '[deleted]' ])].dropna(subset=['selftext'])
    for text in clean['selftext']:
        # turn to lowercase
        text = text.lower()
        # remove punctuation
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        # tokenize
        tokens = word_tokenize(text)
        # lemmatize
        lemmas = ' '.join([wordnet_lemmatizer.lemmatize(token) for token in tokens])
        # save
        total += lemmas
        counter += 1
        if counter % 100 == 0:
            print("Saved " + str(counter) + " out of " + str(dfLength) + " entries")
    return total

#print(string.punctuation)
