import csv
import pandas as pd
import cv2
import glob
import os
import numpy as np
face_id = [] 


for infile in sorted(glob.glob(r'C:\Implementations\Ccollected Images\all_faces/*.png')):
    face_id.append(infile)
face_id1 = sorted(face_id)

directory=face_id1[0][47:]


f1 = open(r"C:\Implementations\Files from Multi_task_CNN\merged\EXPR.txt", "r")
expression=[]
for x in f1:
  expression.append(x)
expression=expression[1:]

f2 = open(r"C:\Implementations\Files from Multi_task_CNN\merged\VA.txt", "r")
VA=[]
for y in f2:
  VA.append(y)
VA=VA[1:]

#### AUs ####
f3 = open(r"C:\Implementations\Files from Multi_task_CNN\merged\AU.txt", "r")
au_=[]
for z in f3:
  au_.append(z)
b=au_[0]
au=au_[1:]


def label_(i):
    directory=face_id1[i][47:]
    directory=directory[:-4]
    expr=int(expression[i])
    parent_dir=r'C:\Implementations\MultiTask CNN Results'+str('/')+str(expr)
   
    path = os.path.join(parent_dir, directory)   
    os.mkdir(path) 
    path_to_csv=path+'/'+directory+'.csv'
    
    c=VA[i][:-1]
    for position,char in enumerate(c):
        if char==',':
            h=position
    c1=c[:h]
    c2=c[h+1:]
    with open(path_to_csv,'a+', newline='') as file:
        writer = csv.writer(file,delimiter=',')
        writer.writerow(["Face_ID:",directory])
        writer.writerow(["Expression:",expr])
        writer.writerow(["Valence:",c1])
        writer.writerow(["Arousal:",c2])
        #writer.writerow([VA])
        
        writer.writerow(["Action Units(AU),0= Absence & 1=Presence"])
             
        a=au[i]
        a_list1 = b.split(',')
        a_list = a.split(',')
        
        writer.writerow(a_list1)
        writer.writerow(a_list)
        
        for j in range(5):
            writer.writerow([''])
        writer.writerow(["Neutral=0"])
        writer.writerow(["Anger=1"])
        writer.writerow(["Disgust=2"])
        writer.writerow(["Fear=3"])
        writer.writerow(["Happiness=4"])
        writer.writerow(["Sadness=5"])
        writer.writerow(["Surprise=6"])
        
    img=cv2.imread(face_id1[i])
    cv2.imwrite(os.path.join(path,directory+".png"),img)
    
    ## oRIGNAL IMAGES
    
    path_=r'C:\Implementations\Ccollected Images\WholeBody'+'/'+directory+'.png'
    print(path_)
    
    img_=cv2.imread(path_)
    cv2.imwrite(os.path.join(path,directory+".jpg"),img_)
    
    file.close()
        
        
for i in range(1855):
    _=label_(i)


        



                