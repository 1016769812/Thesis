import logging
import string
import os
import pickle
from pprint import pprint
import re
import numpy as np
import pandas as pd
from IPython.display import clear_output
from more_itertools import chunked

# NLTK
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Gensim
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import gensim

# SpaCy
import spacy
import en_core_web_sm
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()

import pyLDAvis
import matplotlib.pyplot as plt

# Logging if you want it
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Suppressing it should you want to
logging.getLogger().setLevel(logging.CRITICAL)

# Suppressing warnings
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

#SKLearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text