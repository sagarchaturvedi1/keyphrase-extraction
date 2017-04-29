'''
Created on Mar 3, 2017

@author: sagar
'''

import os
from nltk import word_tokenize,pos_tag
import codecs
import re

def getWordFeatureLine(word):

    def f1(w):return len(w)
    def f2(w):return int(len(w)>0 and ord(w[0])>=ord('A') and ord(w[0])<=ord('Z'))
    def f3(w):
        for c in w:
            if ord(c)>=ord('A') and ord(c)<=ord('Z'):continue
        else:return 0
        return 1

    def f4(w):
        for c in w:
            if ord(c)>=ord('0') and ord(c)<=ord('9'):return 1
        return 0

    def f5(w):
        if len(w)==0:return 0
        return w[-1]

    def f6(w):
        if len(w)<2:return 0
        return w[-2:]

    features=[f1,f2,f3,f4,f5,f6]
    line=''
    for f in features:
        try:
            line+=str(f(word))+' '
        except:
            line+='0 '
    line=line.strip()

    return line

def processAnnLines(lines):
    t=[]
    allowed=['Process','Task','Material']
    for line in lines:
        words=line.strip().split()
        if words[1] not in allowed:continue
        s=''
        for w in words[4:]:
            s+=w+' '
        s=s.strip()
        t.append((words[0],words[1],words[2],words[3],s))
    return t

def getFFLines(ann,txt_line):
    ff=[]
    words_indices=[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', txt_line)]
    words,indices=zip(*words_indices)
    words=list(words)
    tags=pos_tag(words)
    for i in range(len(words)):
        endPunc='0'
        if words[i][-1]=='.' or words[i][-1]==',':
            words[i]=words[i][:-1]
            endPunc='1'
        if indices[i][1]+1<len(txt_line) and (txt_line[indices[i][1]+1]=='.' or txt_line[indices[i][1]+1]==','):
            endPunc='1'
        line=words[i]+' '+tags[i][1]+' '+getWordFeatureLine(words[i])#+' '+endPunc
        index=indices[i]
        index=(index[0],index[1]+1)
        key_phrase='0'
        for t in ann:
            try:
                if index[0]>=int(t[2]) and index[0]<int(t[3]):
                    key_phrase='1'
                    break
            except Exception as e:
                pass
        # line+=' '+key_phrase
        ff.append(line)
    return ff

#os.chdir(r'scienceie2017_dev\dev')
os.chdir(r'C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\data\train')

files=os.listdir('.')

filenames=set()
for name in files:
    filenames.add(name[:name.index('.')])
all_ff=[]
for name in filenames:
    f=open(name+'.ann','r')
    ann_lines=f.readlines()
    f.close()
    f=codecs.open(name+'.txt','r','utf-8')
    txt_line=f.read()
    f.close()
    # txt_lines=processTxtLines(txt_lines)
    ann=processAnnLines(ann_lines)
    # txt_line=txt_lines[0]
    ff=getFFLines(ann,txt_line)
    all_ff+=ff
    f=open(name+'.ffb','w')
    for line in ff:
        f.write(line.encode('utf-8')+'\n')
    f.close()
    print 'Created boundary_ff for',name

os.chdir(r'C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\script\subtaskA')
f=open('all_ff.ffb','w')
for line in all_ff:
    f.write(line.encode('utf-8')+'\n')
f.close()