'''
Created on Mar 26, 2017

@author: sagar
'''

import os
from nltk import word_tokenize
import re
from ff import *
import codecs

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

def processTxtLines(lines):
    new_lines=[]
    for line in lines:
        new_line=re.sub('\[.*?\]','',line).strip()
        new_lines.append(new_line)
    return new_lines

def getFFLines(ann,txt_line):
    ff=[]
    ff_tag=[]
    for t in ann:
        try:
            start=int(t[2])
            end=int(t[3])
            pre_words=['<start>','<start>']+txt_line[:start].split()
            post_words=txt_line[end:].split()+['<end>','<end>']
            words=t[4].split()
            all_words=pre_words+words+post_words
            length=len(words)
            for i in range(len(pre_words),len(pre_words)+length):
                ff_words=(all_words[i-2],all_words[i-1],all_words[i],all_words[i+1],all_words[i+2])
                ff_features=[]
                ff_func=[f1,f2,f3,f4]
                for func in ff_func:
                    output=func(ff_words)
                    if isinstance(output,list) or isinstance(output,tuple):
                        ff_features+=list(output)
                    else:
                        ff_features+=[output]
                ff_tag.append(t[1])
                ff_line_words=list(ff_words)+ff_features#+[t[1]]
                ff_line=u''
                for w in ff_line_words:
                    ff_line+=unicode(w)+u' '
                ff_line=ff_line.strip()
                ff.append(ff_line)
        except:
            pass
    return ff


#os.chdir(r'scienceie2017_dev\dev')
os.chdir(r'C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\data\dev')

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
    f=open(name+'.ff','w')
    for line in ff:
        f.write(line.encode('utf-8')+'\n')
    f.close()
    print 'Created ff for',name

os.chdir(r'C:\Users\sagar\Dropbox\CourseWork\Structured Prediction\Project\work\script\subtaskB')
f=open('all_dev_ff.ff','w')
for line in all_ff:
    f.write(line.encode('utf-8')+'\n')
f.close()