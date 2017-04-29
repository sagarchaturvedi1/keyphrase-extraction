'''
Created on Mar 25, 2017

@author: sagar
'''


# coding: utf-8

# In[112]:

import csv
import spacy
import en_core_web_sm
import pandas as pd

nlp = en_core_web_sm.load()


# In[113]:

train = [line for line in csv.reader(open("train_data_rnn.txt", encoding='utf8'), dialect="excel-tab")]
test = [line for line in csv.reader(open("test_data_rnn.txt", encoding='utf8'), dialect="excel-tab")]


# In[114]:

train[0]


# In[115]:

postags = []
words = []

with open('new_train.txt', 'w', encoding='utf8') as file:
    for index, tup in enumerate(train, start=1):
        tag = ''
        if len(tup) != 2 and len(words) > 0: 
            file.write('\t'.join(words) + '\n')
            words = []
            continue
        if len(tup) != 2: continue
        doc = nlp(tup[0])
        for word in doc:
            tag = word.tag_
           # print(index, word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)
    #     if index % 1000 == 0: print(index)
        postags.append(tag)
        words.append(tup[0] + ' ' + tag + ' ' + tup[1])

words = []

with open('new_test.txt', 'w', encoding='utf8') as file:
    for index, tup in enumerate(test, start=1):
        tag = ''
        if len(tup) != 2 and len(words) > 0: 
            file.write('\t'.join(words) + '\n')
            words = []
            continue
        if len(tup) != 2: continue
        doc = nlp(tup[0])
        for word in doc:
            tag = word.tag_
           # print(index, word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)
    #     if index % 1000 == 0: print(index)
        postags.append(tag)
        words.append(tup[0] + ' ' + tag + ' ' + tup[1])


# In[116]:

with open('new_train.txt', encoding='utf8') as f:
    train = [tuple(i.split('\t')) for i in f]
train = [tuple(tuple(l.split()) for l in tuple(line)) for line in csv.reader(open("new_train.txt", encoding='utf8'), dialect="excel-tab")]
len(train)


# In[ ]:




# In[ ]:



