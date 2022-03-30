import csv
import os
import glob
import cv2
import numpy as np


face_ids=[]
all_Image_ID=[]
for infile in sorted(glob.glob(r'C:\Implementations\Ccollected Images\all_faces/*.png')):
    face_ids.append(infile)
    all_Image_ID.append(infile[47:-4])
    

Neutral = r'C:\Implementations\MultiTask CNN Results\0'
Neutral_Multi_CNN = [name for name in os.listdir(Neutral) if os.path.isdir(os.path.join(Neutral, name))]
    
Angry = r'C:\Implementations\MultiTask CNN Results\1'
Angry_Multi_CNN = [name for name in os.listdir(Angry) if os.path.isdir(os.path.join(Angry, name))]

Disgust=r'C:\Implementations\MultiTask CNN Results\2'
Disgust_Multi_CNN = [name for name in os.listdir(Disgust) if os.path.isdir(os.path.join(Disgust, name))]

Fear = r'C:\Implementations\MultiTask CNN Results\3'
Fear_Multi_CNN = [name for name in os.listdir(Fear) if os.path.isdir(os.path.join(Fear, name))]

Happy=r'C:\Implementations\MultiTask CNN Results\4'
Happy_Multi_CNN = [name for name in os.listdir(Happy) if os.path.isdir(os.path.join(Happy, name))]


Sad=r'C:\Implementations\MultiTask CNN Results\5'
Sad_Multi_CNN = [name for name in os.listdir(Sad) if os.path.isdir(os.path.join(Sad, name))]

Surprise=r'C:\Implementations\MultiTask CNN Results\6'
Surprise_Multi_CNN = [name for name in os.listdir(Surprise) if os.path.isdir(os.path.join(Surprise, name))]



face_id2=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\angry/*.png')):
    face_id2.append(infile)
angry_CNN_Simple=[]
for i in range(len(face_id2)):
    angry_CNN_Simple.append(face_id2[i][60:-4])
    

face_id1_=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\disgust/*.png')):
    face_id1_.append(infile)
Disgust_CNN_Simple=[]
for i in range(len(face_id1_)):
    Disgust_CNN_Simple.append(face_id1_[i][62:-4])

face_id3=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\fear/*.png')):
    face_id3.append(infile)
fear_CNN_Simple=[]
for i in range(len(face_id3)):
    fear_CNN_Simple.append(face_id3[i][59:-4])

face_id4=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\happy/*.png')):
    face_id4.append(infile)
happy_CNN_Simple=[]
for i in range(len(face_id4)):
    happy_CNN_Simple.append(face_id4[i][60:-4])
    
    
face_id1=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\neutral/*.png')):
    face_id1.append(infile)
Neutral_CNN_Simple=[]
for i in range(len(face_id1)):
    Neutral_CNN_Simple.append(face_id1[i][62:-4])
 
face_id5=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\sad/*.png')):
    face_id5.append(infile)
sad_CNN_Simple=[]
for i in range(len(face_id5)):
    sad_CNN_Simple.append(face_id5[i][58:-4])
    
face_id6=[]
for infile in sorted(glob.glob(r'C:\Implementations\SimpleCNN_Emotions+Gender\emotions\surprise/*.png')):
    face_id6.append(infile)
Surprise_CNN_Simple=[]
for i in range(len(face_id6)):
    Surprise_CNN_Simple.append(face_id6[i][58:-4])
    




face_id1=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\angry/*.png')):
    face_id1.append(infile)
Angry_Xception=[]
for i in range(len(face_id1)):
    Angry_Xception.append(face_id1[i][71:-4])

face_id2=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\disgust/*.png')):
    face_id2.append(infile)
Disgust_Xception=[]
for i in range(len(face_id2)):
    Disgust_Xception.append(face_id2[i][73:-4])

face_id3=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\fear/*.png')):
    face_id3.append(infile)
Fear_Xception=[]
for i in range(len(face_id3)):
   Fear_Xception.append(face_id3[i][70:-4])

face_id4=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\happy/*.png')):
    face_id4.append(infile)
Happy_Xception=[]
for i in range(len(face_id4)):
    Happy_Xception.append(face_id4[i][71:-4])
    
face_id5=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\neutral/*.png')):
    face_id5.append(infile)
Neutral_Xception=[]
for i in range(len(face_id5)):
    Neutral_Xception.append(face_id5[i][73:-4])
    
face_id6=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\sad/*.png')):
    face_id6.append(infile)
Sad_Xception=[]
for i in range(len(face_id6)):
    Sad_Xception.append(face_id6[i][69:-4])
    
face_id7=[]
for infile in sorted(glob.glob(r'C:\Implementations\XCeption_Emotions +Gender\emotion_BigXception\surprise/*.png')):
    face_id7.append(infile)
Surprise_Xception=[]
for i in range(len(face_id7)):
    Surprise_Xception.append(face_id7[i][74:-4])
    

### OpneFace ####

Angry = r'C:\Implementations\Emotions from AUs of OpenFace\0'
Angry_OpenFace = [name for name in os.listdir(Angry) if os.path.isdir(os.path.join(Angry, name))]


Disgust=r'C:\Implementations\Emotions from AUs of OpenFace\1'
Disgust_OpenFace = [name for name in os.listdir(Disgust) if os.path.isdir(os.path.join(Disgust, name))]

Fear=r'C:\Implementations\Emotions from AUs of OpenFace\2'
Fear_OpenFace = [name for name in os.listdir(Fear) if os.path.isdir(os.path.join(Fear, name))]

Happy=r'C:\Implementations\Emotions from AUs of OpenFace\3'
Happy_OpenFace = [name for name in os.listdir(Happy) if os.path.isdir(os.path.join(Happy, name))]

Sad=r'C:\Implementations\Emotions from AUs of OpenFace\4'
Sad_OpenFace = [name for name in os.listdir(Sad) if os.path.isdir(os.path.join(Sad, name))]




def search(list, platform):
    for i in range(len(list)):
        if list[i] == platform:
            return True
    return False
TH=1

for i in range(1855):
    
    Image_ID=all_Image_ID[i]
    
    G=np.zeros([7,1])
    
    ### 1-Simple_CNN #####
    _yes1 =search(angry_CNN_Simple,Image_ID)
    if _yes1==True:
        G[0]=TH
        
    _yes2 =search(Disgust_CNN_Simple,Image_ID)
    if _yes2==True:
        G[1]=TH
        
    _yes3 = search(fear_CNN_Simple,Image_ID)
    if _yes3==True:
        G[2]=TH
        
    _yes4 =search(happy_CNN_Simple,Image_ID)
    if _yes4==True:
        G[3]=TH
    _yes4_ =search(Neutral_CNN_Simple,Image_ID)
    if _yes4_==True:
        G[4]=TH
        
    _yes5 =search(sad_CNN_Simple,Image_ID)
    if _yes5==True:
        G[5]=TH
    
    _yes5 =search(Surprise_CNN_Simple,Image_ID)
    if _yes5==True:
        G[6]=TH
        
    ### 2-MultiTask_CNN ###
    _yes1 =search(Angry_Multi_CNN,Image_ID)
    if _yes1==True:
        G[0]=TH
        
    _yes2 =search(Disgust_Multi_CNN,Image_ID)
    if _yes2==True:
        G[1]=TH
        
    _yes3 = search(Fear_Multi_CNN,Image_ID)
    if _yes3==True:
        G[2]=TH
        
    _yes4 =search(Happy_Multi_CNN,Image_ID)
    if _yes4==True:
        G[3]=TH
    _yes5 =search(Neutral_Multi_CNN,Image_ID)
    if _yes5==True:
        G[4]=TH
    
    _yes4 =search(Sad_Multi_CNN,Image_ID)
    if _yes4==True:
        G[5]=TH
    _yes5 =search(Surprise_Multi_CNN,Image_ID)
    if _yes5==True:
        G[6]=TH
    
    
    ## 3-Xceptionnet
    _yes1 =search(Angry_Xception,Image_ID)
    if _yes1==True:
        G[0]=TH
        
    _yes2 =search(Disgust_Xception,Image_ID)
    if _yes2==True:
        G[1]=TH
        
    _yes3 = search(Fear_Xception,Image_ID)
    if _yes3==True:
        G[2]=TH
        
    _yes4 =search(Happy_Xception,Image_ID)
    if _yes4==True:
        G[3]=TH
        
    _yes5 =search(Neutral_Xception,Image_ID)
    if _yes5==True:
        G[4]=TH
        
    _yes6 =search(Sad_Xception,Image_ID)
    if _yes6==True:
        G[5]=TH
        
    _yes7 =search(Surprise_Xception,Image_ID)
    if _yes7==True:
        G[6]=TH
    
    
    ### 4-OpenFace ####
    
    _yes1 =search(Angry_OpenFace,Image_ID)
    if _yes1==True:
        G[0]=TH
        
    _yes2 =search(Disgust_OpenFace,Image_ID)
    if _yes2==True:
        G[1]=TH
        
    _yes3 = search(Fear_OpenFace,Image_ID)
    if _yes3==True:
        G[2]=TH
        
    _yes4 =search(Happy_OpenFace,Image_ID)
    if _yes4==True:
        G[3]=TH
    _yes5 =search(Sad_OpenFace,Image_ID)
    if _yes5==True:
        G[5]=TH
        
    ## genrating the text file 
    path=r'C:\Implementations\Multi_Label/'+str(Image_ID)+'.txt'
    with open(path, 'w') as f:
        for item in G:
            #f.write(str(item) + "\n")
            y=' '.join(map(str, item))
            f.write(f"{y}\n")
     
        
import os
import glob
import numpy as np


text_path=r'C:\Implementations\Multi_Label/'
def READ_DAT():
    image_w=224
    image_h=224
    img_ = [] 
    text_=[]
    for i in range(1384):
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
    
images,text_files=READ_DAT()














    
