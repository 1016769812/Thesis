import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def readfile(filename):
    with open(filename,'r', encoding='utf-8') as f:
        return f.read()

test=readfile(r"C:\Users\Li\Desktop\with_chinese.txt")

stop = set(stopwords.words('english') + ['’', '“', '”', 'nbsp', 'http','?'])


def remove_stopwords(texts):
    text = texts.lower()
    return [[word for word in doc if word not in stop] for doc in texts]

texts1=remove_stopwords(test)
print(texts1)

'''counts = {}
for word in texts1:
    counts[word] = counts.get(word,0) + 1

items = list(counts.items())


items.sort(key=lambda x:x[1],reverse=True)
for i in range(50):
    word,count = items[i]
    print("{0:<10},{1:>5}".format(word,count))'''
