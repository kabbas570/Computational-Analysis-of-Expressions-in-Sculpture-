import csv
import os
import glob
import cv2
import numpy as np


face_ids=[]
all_Image_ID=[]
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/DeepLearning/whole body/*.png')):
    face_ids.append(infile)
    all_Image_ID.append(infile[52:-4]) ## 56 for faces and 52 for whole body

print(len(all_Image_ID))
text_path='/home/user01/data_ssd/Abbas/DeepLearning/text_files1/'
u=text_path+all_Image_ID[0]+'.txt'
print(u)

def READ_DAT():
    image_w=224
    image_h=224
    img_ = [] 
    text_=[]
    for i in range(1855):
        img=cv2.imread(face_ids[i])
        Image_ID=all_Image_ID[i]
        img = cv2.resize (img, (image_w,image_h), interpolation = cv2.INTER_AREA) 
        img=img/255
        
        img_.append(img)
        
        u=text_path+Image_ID+'.txt'
        with open(u) as f:
            lines = f.readlines()
            A=[float(k) for k in lines]
            text_.append(A)
            
            #for line in f:
                #if line: #avoid blank lines
                    #text_.append(float(line.strip()))
            
            
    img_ = np.array(img_)
    text_ = np.array(text_)
    
    return img_,text_