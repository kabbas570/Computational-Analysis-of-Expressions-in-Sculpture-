import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.cm as cm
import tensorflow.keras as keras
from tensorflow.keras import optimizers

from sklearn.utils import shuffle

from Read_Data import READ_DAT
images,text_files=READ_DAT()

images_s, text_s = shuffle(images,text_files, random_state=2009)
del images
del text_files

txt_test=text_s[0:370]
images_test=images_s[0:370]

images_val=images_s[370:470]
txt_val=text_s[370:470]

images_Training=images_s[470:1855]
txt_Training=text_s[470:1855]

del images_s
del text_s

call_back=tf.keras.callbacks.EarlyStopping(
       monitor="loss",
      patience=10,
       mode="min")
def Hamming_loss(y_true, y_pred):
       tmp = K.abs(y_true-y_pred)
       return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))
       
### Model###    
for i in  range(5):
  model = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3),include_top=False,weights='imagenet')
  x=keras.layers.GlobalAveragePooling2D()(model.output)
  x = keras.layers.Dense(128, activation='relu')(x)
  output = keras.layers.Dense(7, activation='sigmoid')(x)
  model = keras.models.Model(inputs=model.input, outputs=output) 
  #model.summary() 
     
  Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
  model.compile(optimizer=Adam, loss='binary_crossentropy',metrics=[Hamming_loss])
  history =model.fit(images_Training,txt_Training, batch_size=5, validation_data=(images_val, txt_val),
                       epochs=200,callbacks=[call_back])
                   
  model.save_weights('VGG16_'+str(i+1)+'.h5')


from sklearn.metrics import classification_report

txt_test[:,1]=txt_test[:,1]*2
txt_test[:,2]=txt_test[:,2]*3
txt_test[:,3]=txt_test[:,3]*4
txt_test[:,4]=txt_test[:,4]*5
txt_test[:,5]=txt_test[:,5]*6
txt_test[:,6]=txt_test[:,6]*7

crp_reports=[]
for i in range(5):
  model.load_weights('VGG16_'+str(i+1)+'.h5')
  result_1 = model.predict(images_test)
  result_N=np.zeros([result_1.shape[0],result_1.shape[1]])
  result_N[np.where(result_1>0.5)]=1
  result_N[:,1]=result_N[:,1]*2
  result_N[:,2]=result_N[:,2]*3
  result_N[:,3]=result_N[:,3]*4
  result_N[:,4]=result_N[:,4]*5
  result_N[:,5]=result_N[:,5]*6
  result_N[:,6]=result_N[:,6]*7
  G=np.reshape(txt_test,(txt_test.shape[0]*txt_test.shape[1],1))
  G=G.astype(int)
  P=np.reshape(result_N,(result_1.shape[0]*result_1.shape[1],1))
  P=P.astype(int)
  crp1=classification_report(G, P, labels=[1,2,3,4,5,6,7],output_dict=  True)
  
  crp_reports.append(crp1)

classes=7
keys=['precision','recall','f1-score']
key1={'macro avg','micro avg','weighted avg'}

key_=['1','2','3','4','5','6','7','macro avg','micro avg','weighted avg']

d = {}
for i in range(classes):
    d[str(i+1)] = {}
for j in key1:
    d[j] = {}

for i in key_:   
   temp1 = crp_reports[0][i]
   temp2 = crp_reports[1][i]
   temp3 = crp_reports[2][i]
   temp4 = crp_reports[3][i]
   temp5 = crp_reports[4][i]
   
   for key in keys:
       value_=temp1[key]+temp2[key]+temp3[key]+temp4[key]+temp5[key]
       d[i][key]=value_/5

import pandas as pd
repdf = pd.DataFrame(d).round(2).transpose()
repdf.insert(loc=0, column='class', value=repdf.index)
repdf.to_csv("VGG16_.csv")
