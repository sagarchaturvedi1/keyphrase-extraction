'''
Created on Apr 25, 2017

@author: sagar
'''

import os,re
import codecs
from nltk.tokenize import word_tokenize

def spans(txt):
    tokens= word_tokenize(txt)
    offset = 0
    listm = []
    for token in tokens:
        offset = txt.find(token, offset)
        lm=[]
        lm.append(token)
        lm.append(offset)
        lm.append(offset+len(token))
        listm.append(lm)
        offset += len(token)
     
    return listm   


directory = r"C:/Users/sagar/Dropbox/CourseWork/Structured Prediction/Project/work/rnn code/trainann_tagged/"
files = os.listdir(directory)
fwrite = codecs.open(r"C:/Users/sagar/Dropbox/CourseWork/Structured Prediction/Project/work/rnn code/tagged_text/train_data_rnn.txt", "w+",encoding='utf-8')
l = []

for file in files:
    
    ff = codecs.open(directory+file, "r",encoding='utf-8')
    contents = ff.readlines()
    ftext = codecs.open(r"C:/Users/sagar/Dropbox/CourseWork/Structured Prediction/Project/work/data/train/" + file.replace(".ann",".txt"), "r",encoding='utf-8').read()
    
    try:
        sp = spans(ftext)
    except:
        for token in spans:
            print(token)
        print('Error in tokenization in file',file)
           
    loc_dict={}

    for annotation in contents:
        annotation = annotation.split("||")
        key = annotation[1].split("\n")[0]
        start=(annotation[0].split(" ")[2])
        end=(annotation[0].split(" ")[3])
        loc_dict[int(start)] = end+"||"+" ".join(annotation[0].split()[4:])
    
    end=0
    for token in sp:
        if token[2] > end:
            if token[1] in loc_dict.keys():
            #if token[1] in loc_dict.keys() and 'TSK' in loc_dict[token[1]]:
                values=loc_dict[token[1]].split("||")[1].split(' ')
                for i in values:
                    l.append(i.replace('|','\t'))
                end=int(loc_dict[token[1]].split("||")[0])
            else:
                if bool(re.search('[a-zA-Z0-9]', token[0], re.IGNORECASE)):
                    l.append(token[0]+'\t'+'O')
    l.append("\n")

fwrite.write("\n".join(l))
 

#===============================================================================
# for item in l:
#     fwrite.write(item)   
#===============================================================================
    
        
               
        
    
     
