import csv
import pandas as pd
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


face_id = [] 
csv_id = []
imgs_id=[]
csv_path=r'C:\Implementations\OpenFace\csv files'
img_path=r'C:\Implementations\OpenFace\images'
for infile in sorted(glob.glob('C:/Implementations/OpenFace/faces/*.png')): # path to masks of train data
    face_id.append(infile)
    csv_id.append(csv_path+'/'+infile[34:-4]+'.csv')
    imgs_id.append(img_path+'/'+infile[34:-4]+'.jpg')
    

to_save_path=r'C:\Implementations\Emotions from AUs of OpenFace'

Image_ID=face_id[100][34:-4] # FitzWilliam=61,Maa=60, Palace=63, BM=52,NM=73, Buddha=56Image_ID=face_id[i][56:-28] # FitzWilliam=61,Maa=60, Palace=63, BM=52,NM=73, Buddha=56




ANGER=['04','05','07','10','23','25','26','17']
DISGUST=['09','10','25','26','17',]
FEAR=['01','02','04','05','20','25','26']
HAPPY=['06','12']
SADNESS=['01','04','15','06','17']
SURPRISE=['01','02','05','26']

def intensity_index(i):
    Expression=[]
    data= pd.read_csv(csv_id[i])
    AU_intensities1 = data.iloc[:, 676:693]
    AU_intensities = AU_intensities1.drop(AU_intensities1.columns[[9, 16]], axis=1)
    
    k=0
    if (csv_id[i]==csv_id[i-1]):
        k=k+1
    if (csv_id[i]==csv_id[i-2]):
        k=k+1
    if (csv_id[i]==csv_id[i-3]):
        k=k+1
    if (csv_id[i]==csv_id[i-4]):
        k=k+1
    if (csv_id[i]==csv_id[i-5]):
        k=k+1
    if (csv_id[i]==csv_id[i-6]):
        k=k+1
    if (csv_id[i]==csv_id[i-7]):
        k=k+1
        
    AU_intensities_Not0=AU_intensities.columns[(AU_intensities != 0).iloc[k]] # indices where column is not 0.

    au=[]   # extract only AUs numbers only i.i ,'01'.'02','05'
    for j in range(len(AU_intensities_Not0)):
        c=AU_intensities_Not0[j][3:-2]
        au.append(c)
    ANGER_=set(au) & set(ANGER)
    DISGUST_=set(au) & set(DISGUST)
    FEAR_=set(au) & set(FEAR)
    HAPPY_=set(au) & set(HAPPY)
    SADNESS_=set(au) & set(SADNESS)
    SURPRISE_=set(au) & set(SURPRISE)
    
    Anger=len(ANGER_)
    Disgust=len(DISGUST_)
    Fear=len(FEAR_)
    Happy=len(HAPPY_)
    Sadness=len(SADNESS_)
    Surprise=len(SURPRISE_)
    
    Expression.append(Anger)
    Expression.append(Disgust)
    Expression.append(Fear)
    Expression.append(Happy)
    Expression.append(Sadness)
    Expression.append(Surprise)
    max_value = max(Expression)
    indices_intensity = [index for index, value in enumerate(Expression) if value == max_value]
    return indices_intensity,AU_intensities1

def presence_index(i):
    Expression=[]
    
    data= pd.read_csv(csv_id[i])
    AU_presense1 = data.iloc[:, 693:711]
    AU_presense = AU_presense1.drop(AU_presense1.columns[[9, 16,17]], axis=1)
    
    k=0
    if (csv_id[i]==csv_id[i-1]):
        k=k+1
    if (csv_id[i]==csv_id[i-2]):
        k=k+1
    if (csv_id[i]==csv_id[i-3]):
        k=k+1
    if (csv_id[i]==csv_id[i-4]):
        k=k+1
    if (csv_id[i]==csv_id[i-5]):
        k=k+1
    if (csv_id[i]==csv_id[i-6]):
        k=k+1
    if (csv_id[i]==csv_id[i-7]):
        k=k+1
    col=AU_presense.columns[(AU_presense == 1).iloc[k]] # indices where column is 1.

    au=[]   # extract only AUs numbers only i.i ,'01'.'02','05'
    for j in range(len(col)):
        c=col[j][3:-2]
        au.append(c)
        
    ANGER_=set(au) & set(ANGER)
    DISGUST_=set(au) & set(DISGUST)
    FEAR_=set(au) & set(FEAR)
    HAPPY_=set(au) & set(HAPPY)
    SADNESS_=set(au) & set(SADNESS)
    SURPRISE_=set(au) & set(SURPRISE)
    
    Anger=len(ANGER_)
    Disgust=len(DISGUST_)
    Fear=len(FEAR_)
    Happy=len(HAPPY_)
    Sadness=len(SADNESS_)
    Surprise=len(SURPRISE_)
    
    Expression.append(Anger)
    Expression.append(Disgust)
    Expression.append(Fear)
    Expression.append(Happy)
    Expression.append(Sadness)
    Expression.append(Surprise)
    max_value = max(Expression)
    indices_presence = [index for index, value in enumerate(Expression) if value == max_value]
    return indices_presence,AU_presense1

#path=r'C:\DATASET_COLLECTED\LABELLED\FitzWilliam\Both'
for i in range(len(face_id)):
    indices_I,AU_intensities1=intensity_index(i)
    indices_P,AU_presense1=presence_index(i)
    tie_= set(indices_I) & set(indices_P)
    L=len(tie_)
    
    if L==0:
        tie_='Undecided'
        parent_dir = to_save_path+str('/')+str(-1)
    if L>1:
        parent_dir = to_save_path+str('/')+'00'
    if L==1:
        parent_dir = to_save_path+str('/')+str(tie_)[1:-1]
    
    #print(L)
    #print(parent_dir)
    Image_ID=face_id[i][34:-4] # FitzWilliam=61,Maa=60, Palace=63, BM=52,NM=73, Buddha=56
    #print(Image_ID)
    img=cv2.imread(face_id[i])    
    img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
    
    directory = Image_ID
    Image_ID_org=Image_ID
    print(Image_ID)
    if (csv_id[i]==csv_id[i-1]):
        directory= Image_ID + str('_2')
        Image_ID= Image_ID + str('_2')
    if (csv_id[i]==csv_id[i-2]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_3')
        Image_ID= Image_ID_ + str('_3')
    if (csv_id[i]==csv_id[i-3]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_4')
        Image_ID= Image_ID_ + str('_4')
    if (csv_id[i]==csv_id[i-4]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_5')
        Image_ID= Image_ID_ + str('_5')
    if (csv_id[i]==csv_id[i-5]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_6')
        Image_ID= Image_ID_ + str('_6')
    if (csv_id[i]==csv_id[i-6]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_7')
        Image_ID= Image_ID_ + str('_7')
    if (csv_id[i]==csv_id[i-7]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_8')
        Image_ID= Image_ID_ + str('_8')
    if (csv_id[i]==csv_id[i-8]):
        Image_ID_= Image_ID_org
        directory= Image_ID_ + str('_9')
        Image_ID= Image_ID_ + str('_9')
    
    
    # Parent Directory path
    #parent_dir = "C:\DATASET_COLLECTED\DataLabelledByOpenFace"+str(Expression_is)
      
    # Path
    path = os.path.join(parent_dir, directory)
    
    os.mkdir(path)
    #print("Directory '% s' created" % directory)
    
    path_to_csv=path+'/'+Image_ID+'.csv'
    with open(path_to_csv,'w', newline='') as file:
        writer = csv.writer(file,delimiter=',')
        #writer.writerow([AU_presense1])
        writer.writerow([" Presence - if AU is visible in the face"])
        AU_presense1.to_csv(file, encoding='utf-8', index=False)
        writer.writerow([''])
        writer.writerow([" Intensity - how intense is the AU (minimal to maximal) on a 5 point scalee"])
        AU_intensities1.to_csv(file, encoding='utf-8', index=False)
        writer.writerow([''])
        writer.writerow([" Predicted emotion based on Presence,Intensity, and comparison of Both "])
        writer.writerow([''])
        writer.writerow(["Face_ID:",Image_ID])

        writer.writerow([str(indices_P)+str(':'),"Emotion/s predicted based on Presence of AU"])
        writer.writerow([str(indices_I)+str(':'),"Emotion/s predicted based on Intensity of AU"])
        writer.writerow([str(tie_)+str(':'),"Emotion/s predicted based on Intersection of Both"])
        #writer.writerow(["Face_ID:","Emotion@Presence","Emotion@Intensity","Emotion@Combined"])
        #writer.writerow([Image_ID,indices_P,indices_I,tie_])
        for j in range(5):
            writer.writerow([''])
        writer.writerow(["Anger=0"])
        writer.writerow(["Disgust=1"])
        writer.writerow(["Fear=2"])
        writer.writerow(["Happy=3"])
        writer.writerow(["Sadness=4"])
        writer.writerow(["Surprise=5"])
        file.close()

    cv2.imwrite(os.path.join(path,Image_ID+".png"),img)
    print(i)
    ### Reading the Orignal image ###
    org_img=cv2.imread(imgs_id[i])
    cv2.imwrite(os.path.join(path,Image_ID+".jpg"),org_img)