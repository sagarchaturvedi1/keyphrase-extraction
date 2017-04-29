
# coding: utf-8

# In[136]:

from itertools import chain
#import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import unicodecsv as csv


train_sents = [tuple(tuple(l.split()) for l in tuple(line)) for line in csv.reader(open(r"C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\rnn code\tagged_text\crf_train.txt"), dialect="excel-tab")]
test_sents = [tuple(tuple(l.split()) for l in tuple(line)) for line in csv.reader(open(r"C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\rnn code\tagged_text\crf_test.txt"), dialect="excel-tab")]

# Data format:

# In[139]:

train_sents[0]


# ## Features


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
#         'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
#             '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
#             '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
        
#===============================================================================
#     if i > 1:
#         word1 = sent[i-2][0]
#         postag1 = sent[i-2][1]
#         features.extend([
#             '-2:word.lower=' + word1.lower(),
#             '-2:word.istitle=%s' % word1.istitle(),
#             '-2:word.isupper=%s' % word1.isupper(),
#             '-2:postag=' + postag1,
# #             '-1:postag[:2]=' + postag1[:2],
#         ])
#     else:
#          features.append('BOS')
#         
#     if i < len(sent)-2:
#         word1 = sent[i+2][0]
#         postag1 = sent[i+2][1]
#         features.extend([
#             '+2:word.lower=' + word1.lower(),
#             '+2:word.istitle=%s' % word1.istitle(),
#             '+2:word.isupper=%s' % word1.isupper(),
#             '+2:postag=' + postag1,
# #             '+1:postag[:2]=' + postag1[:2],
#         ])
#     else:
#         features.append('EOS')
#===============================================================================
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]    


sent2features(train_sents[0])[0]



X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]



X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    try:
        trainer.append(xseq, yseq)
    except:
        #print('1')
        pass

#trainer.select(algorithm='l2sgd')



trainer.set_params({
    'c1': 5.0,   # coefficient for L1 penalty
    'c2': 0.0,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


# Possible parameters for the default training algorithm:

# In[145]:

trainer.params()


# Train the model:

# In[146]:

# %%time
trainer.train('keyphraseB.crfsuite')



trainer.logparser.last_iteration




tagger = pycrfsuite.Tagger()
tagger.open('keyphraseB.crfsuite')


# Let's tag a sentence to see how it works:

# In[151]:

example_sent = test_sents[0]
#print(' '.join(sent2tokens(example_sent)), end='\n\n')

#print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
#print("Correct:  ", ' '.join(sent2labels(example_sent)))




def classify(y_true, y_pred):
    
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

y_pred = [tagger.tag(xseq) for xseq in X_test]


print(classify(y_test, y_pred))


from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


# We can see that, for example, it is very likely that the beginning of an organization name (B-ORG) will be followed by a token inside organization name (I-ORG), but transitions to I-ORG from tokens with other labels are penalized. Also note I-PER -> B-LOC transition: a positive weight means that model thinks that a person name is often followed by a location.
# 
# Check the state features:

# In[156]:

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])






