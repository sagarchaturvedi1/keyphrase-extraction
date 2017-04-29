'''
Created on Apr 27, 2017

@author: sagar
'''
import sys,os,re




def tagann():
    directory=r"C:/Users/sagar/Dropbox/CourseWork/Structured Prediction/Project/work/data/test/"
    files=os.listdir(directory)
    flag=1
    for file in files:
        if file.endswith(".ann"):
            fread=open(directory+file,"U")
            lines=fread.readlines()
            fwrite = open(directory.replace('test',"testann_tagged") + file, "w+")
            for line in lines:
                line=re.sub(" +"," ",line)
    
                line=line.split()
                templine=" ".join(line[4:])
                if line[1]=="Process":
                            line[4]=line[4]+"|B-PROC"
                            if len(line[4:]) != 1:
                                line[len(line)-1]=line[len(line)-1]+"|L-PROC"
                            for i in range(5,len(line)-1):
                                line[i] = line[i] + "|I-PROC"
    
                elif line[1]=="Material":
                            line[4]=line[4]+"|B-MAT"
                            if len(line[4:])!=1:
                                line[len(line)-1]=line[len(line)-1]+"|L-MAT"
                            for i in range(5,len(line)-1):
                                line[i] = line[i] + "|I-MAT"
                elif line[1] == "Task":
                        line[4] = line[4] + "|B-TSK"
                        if len(line[4:]) != 1:
                            line[len(line) - 1] = line[len(line) - 1] + "|L-TSK"
                        for i in range(5, len(line) - 1):
                            line[i] = line[i] + "|I-TSK"
                else:
                        continue
                print>>fwrite," ".join(line)+"||"+templine
    


tagann()