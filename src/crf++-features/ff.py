'''
Created on Apr 20, 2017

@author: sagar
'''
from nltk import pos_tag,word_tokenize

def f1(w):                        # FirstCap
    if ord(w[2][0])>=ord('A') and ord(w[2][0])<=ord('Z'):
        return 1
    return 0

def f2(w):                        # AllCaps
    for c in w[2]:
        if ord(c)>=ord('A') and ord(c)<=ord('Z'):
            continue
        else:
            return 0
    return 1

def f3(w):                        # len
    return len(w[2])

def f4(w):
    tags=pos_tag(list(w))
    _,tags=zip(*tags)
    return tags