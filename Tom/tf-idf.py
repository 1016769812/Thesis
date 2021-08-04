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

document1 = """Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen."""

document2 = """Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known."""

document3 = """Monty Python (also collectively known as the Pythons) are a British 
surreal comedy group who created the sketch comedy television show Monty Python's 
Flying Circus, which first aired on the BBC in 1969. Forty-five episodes were made 
over four series."""

document4 = """Python is an interpreted, high-level, general-purpose programming language. 
Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes 
code readability with its notable use of significant whitespace. Its language constructs and 
object-oriented approach aim to help programmers write clear, logical code for small and 
large-scale projects."""

document5 = """The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment."""

document6 = """The Pythonidae, commonly known simply as pythons, from the Greek word python 
(πυθων), are a family of nonvenomous snakes found in Africa, Asia, and Australia. 
Among its members are some of the largest snakes in the world. Eight genera and 31
species are currently recognized."""

test_list = [document1, document2, document3, document4, document5, document6]


# settings that you use for count vectorizer will go here
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, decode_error='ignore', stop_words='english',smooth_idf=True,use_idf=True)

# fit and transform the texts
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(test_list)
print(
tfidf_vectorizer_vectors.toarray()[2])
# get the vector for the third document
vector_tfidfvectorizer = tfidf_vectorizer_vectors[2] # Note that 2 refers to document3, due to zero-based indexing

# place tf-idf values in a DataFrame
df = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print(df.sort_values(by=["tfidf"],ascending=False)[:10])

''' 文档3里前十个最特别的词
               tfidf
comedy        0.424705
monty         0.424705
circus        0.212353
television    0.212353
surreal       0.212353
group         0.212353
collectively  0.212353
aired         0.212353
sketch        0.212353
british       0.212353'''

reddit_list = [trp_pp, sed_pp, mgtow_pp]