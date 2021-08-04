from nltk.tokenize import sent_tokenize
import string
import nltk
import pandas as pd

def readfile(filename):
    with open(filename,'r', encoding='utf-8') as f:
        return f.read()
test=readfile(r"C:\Users\Li\Desktop\test.txt")

sent=sent_tokenize(test)
#print(sent[:10])
'''for i in sent[:10]:
    print(i)'''

def nltk_tokenizer(text):
    text=text.lower()
    tokens=nltk.word_tokenize(text)
    return tokens


def typeTokenRatio(tokens):
    numTokens=len(tokens)
    numTypes=len(tokens)
    return numTypes/numTokens


for row in dfNew['body']:
    tokens=nltk_tokenizer(row)
    print(typeTokenRatio(tokens))
