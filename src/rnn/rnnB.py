'''
Created on Mar 24, 2017

@author: sagar
'''

from __future__ import absolute_import
from __future__ import print_function

from datetime import datetime
import itertools
import os
from pprint import pprint
import sys

from keras import callbacks
from keras.layers import recurrent
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

import numpy as np


np.random.seed(1337)  



def measure(predict, groundtruth, vocab_label_size, bgn_label_idx):
    '''
        get precision, recall, f1 score
    '''
    tp = []
    fp = []
    fn = []
    recall = 0
    precision = 0
    #print((vocab_label_size))
    for i in range(vocab_label_size):
        tp.append(0)
        fp.append(0)
        fn.append(0)

    for i in range(len(groundtruth)):
        if groundtruth[i] == predict[i]:
            tp[groundtruth[i]] += 1

        else:
            fp[predict[i]] += 1
            fn[groundtruth[i]] += 1

    for i in range(vocab_label_size):
        # do not count begin label
        if i == bgn_label_idx:
            continue
        if tp[i] + fp[i] == 0:
            precision += 1
        else:
            precision += float(tp[i]) / float(tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall += 1
        else:
            recall += float(tp[i]) / float(tp[i] + fn[i])

    precision /= (vocab_label_size - 1)
    recall /= (vocab_label_size - 1)
    #pprint(tp)
    #pprint(fp)
    #pprint(fn)
    f1 = 2 * float(precision) * float(recall) / (precision + recall)
    print ('precision: %f, recall: %f, f1 score on testa is %f' % (precision, recall, f1))

def add_begin_end_word(sentences, labels, ngram):
    '''
        adds begin and end words for each sentence
        for bi-directional rnn, we need end words
        '__BGN__' for begin
        '__END__' for end
    '''
    BEGIN_WORD = '__BGN__'
    BEGIN_LABEL = '__BGN__'

    for i in range(len(sentences)):
        sentences[i] = [BEGIN_WORD] * (ngram - 1) + sentences[i]
        labels[i] = [BEGIN_LABEL] * (ngram - 1) + labels[i]


def load_ner_data(fname):
    '''
        load ner data into arrays
    '''
    sentences = []
    labels = []
    with open(fname, encoding="utf8") as fin:
        sentence = []
        label = []
        for l in fin.readlines():
            l = l.strip()
            #print(l)
            if l == '': # sentence ends
                sentences.append(sentence)
                labels.append(label)

                sentence = []
                label = []
            else:
                try:
                    w, l = l.split("\t")
                    sentence.append(w)
                    label.append(l)
                except:
                    pass
                

        if len(sentence) != 0:
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels


def to_ngrams(sentences, labels, ngram):
    '''
        truncates sentences into sub-sentences
    '''
    ngrams = []
    ngram_labels = []

    for s, l in zip(sentences, labels):
        for i in range(len(s) - ngram + 1):
            ngrams.append(s[i : i + ngram])
            ngram_labels.append(l[i + ngram - 1])

    return ngrams, ngram_labels

def to_vocabulary(lst_of_lst):
    # use from_iterable to flat the lst
    return set(itertools.chain.from_iterable(lst_of_lst))


def to_vocab_index(vocab_index_dict, ngrams):
    ngrams_index = []
    for ngram in ngrams:
        ngrams_index.append([vocab_index_dict[w] for w in ngram])
    return ngrams_index


def to_label_index(label_index_dict, labels):
    return [label_index_dict[l] for l in labels]


def get_data(path, ngram, begin_end_word = True):
    '''
        converts conll 2002 dataset into Keras support type
    '''
    # read data into arrays (array of arrays)

    train_sentences, train_labels = load_ner_data(r'C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\rnn code\tagged_text\train_data_rnn.txt')
    testa_sentences, testa_labels = load_ner_data(r'C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\rnn code\tagged_text\test_data_rnn.txt')
    #===========================================================================
    # X_train, X_test, y_train, y_test = train_test_split(train_sentences, train_labels, test_size=0.33, random_state=42)
    # train_sentences, train_labels=X_train,y_train
    # testa_sentences, testa_labels = X_test,y_test
    #===========================================================================

    # add begin and end words into each sentences (the amount of b&e words depends on ngram)
    #   e.g. when ngram == 3
    #   This is an example ---> __BGN__ __BGN__ This is an example __END__ __END__
    if begin_end_word:
        add_begin_end_word(train_sentences, train_labels, ngram)
        add_begin_end_word(testa_sentences, testa_labels, ngram)


    # truncate sentences into ngram sub-sentences, and store the words' number in dictionary into train_x/test_x
    #   e.g. when ngram == 3
    #   __BGN__ __BGN__ This is an example __END__ __END__
    #   become
    #   #__BGN__ #__BGN__ #This
    #   #__BGN__ #This #is
    #   #This #is #an
    #   #is #an #example
    #   #an #example #__END__
    #   #example #__END__ #__END__
    #   where '#' represents a word's number in dictionary
    train_x, train_y = to_ngrams(train_sentences, train_labels, ngram)
    testa_x, testa_y = to_ngrams(testa_sentences, testa_labels, ngram)

    #print(train_x[4])
    #print(train_y[4])

    # generate vocabulary for x and y
    vocab = to_vocabulary(train_sentences) | to_vocabulary(testa_sentences)
    vocab_label = to_vocabulary(train_labels)

    # generate a dictionary mapped from word / label to its index
    vocab_index_dict = {w: i for i, w in enumerate(vocab)}
    label_index_dict = {l: i for i, l in enumerate(vocab_label)}
    #===========================================================================
    # for k,v in label_index_dict.items():
    #     print(k,v)
    #===========================================================================

    # convert from words to words' indexes
    train_x = to_vocab_index(vocab_index_dict, train_x)
    testa_x = to_vocab_index(vocab_index_dict, testa_x)


    # convert from labels to labels' indexes
    train_y = to_label_index(label_index_dict, train_y)
    testa_y = to_label_index(label_index_dict, testa_y)

    bgn_label_idx = label_index_dict['__BGN__']
    testb_x=""
    testb_y=""
    #print(train_x[0])
    return train_x, train_y, testa_x, testb_x, testa_y, testb_y, vocab, vocab_label, bgn_label_idx

def get_nets(name):
    if name=='LSTM':
        return recurrent.LSTM
    elif name=='GRU':
        return recurrent.GRU
    else:
        return recurrent.SimpleRNN

networks = ['SimpleRNN','LSTM','GRU']
epoch_list = [10,35]
ngram_types = [3,5]
batch_sizes = [128,256]
embedding_size = 256
hidden_size = 256

print()
print()
for network in networks:
    for ngram in ngram_types:
        for batch_size in batch_sizes:
            for epochs in epoch_list:
                rnn = get_nets(network)
                
                print('***** Experimenting with rnn type', rnn,', ngrams=',ngram,', batch_size=', batch_size,', epochs = ',epochs,'*****' )
                
                t1 = datetime.now()
                
                
                train_x, train_y, testa_x, testb_x, testa_y, testb_y, vocab, vocab_label, bgn_label_idx = get_data('', ngram)
                
                
                
                vocab_size = len(vocab) + 1
                vocab_label_size = len(vocab_label)
                #print('vocab_size = ', vocab_size)
                #print('vocab_label_size =', vocab_label_size)
                
                # pad x
                #print('Padding train data...')
                train_x_pad = sequence.pad_sequences(train_x, maxlen=ngram)
                #print('Padding test data a...')
                testa_x_pad = sequence.pad_sequences(testa_x, maxlen=ngram)
                
                #print('train_x.shape = {}'.format(train_x_pad.shape))
                #print('testa_x.shape = {}'.format(testa_x_pad.shape))
                
                # convert class vectors to binary class matrices
                train_y_categorical = np_utils.to_categorical(train_y, vocab_label_size)
                testa_y_categorical = np_utils.to_categorical(testa_y, vocab_label_size)
                #testb_y_categorical = np_utils.to_categorical(testb_y, vocab_label_size)
                
                #print('train_y_categorical.shape = {}'.format(train_y_categorical.shape))
                #print('testa_y_categorical.shape = {}'.format(testa_y_categorical.shape))
                #print('testb_y_categorical.shape = {}'.format(testb_y_categorical.shape))
                
                # build model
                print('Build model...')
                model = Sequential()
                model.add(Embedding(vocab_size, embedding_size, mask_zero=True))
                model.add(rnn(hidden_size, return_sequences=False))
                model.add(Dropout(0.5))
                model.add(Dense(vocab_label_size))
                model.add(Activation('softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
                
                # train & test model
                callback = callbacks.TensorBoard(log_dir='C:/Users/sagar/Dropbox/CourseWork/Structured Prediction/Project/work/rnn code/logs', histogram_freq=0, write_graph=True, write_images=False)
                model.fit(train_x_pad, train_y_categorical, batch_size=batch_size, nb_epoch=epochs, callbacks=[callback] ,verbose=0)
                testa_y_predict = model.predict_classes(testa_x_pad, batch_size=batch_size,verbose=0)
                
                # get precision, recall, f1 score
                print('testing test data ...')
                measure(testa_y_predict, testa_y, vocab_label_size, bgn_label_idx)
                t2 = datetime.now()
                print("Total time takes : ",t2-t1)
                print()
                print()
