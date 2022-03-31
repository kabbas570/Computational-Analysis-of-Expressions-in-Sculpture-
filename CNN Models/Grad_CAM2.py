import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
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

def get_img_array(img_path):
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize (img, (224,224), interpolation = cv2.INTER_AREA) 
    img=img/255
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    return img
    

last_conv_layer_name = "conv_pw_13_relu"

# Make model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D,GlobalMaxPooling2D, Reshape,Add, Conv2DTranspose,Activation, Dense, UpSampling2D, SeparableConv2D, BatchNormalization,GlobalAveragePooling2D,Dropout,Flatten
from tensorflow.keras.regularizers import l2, l1
from keras.models import Model
import tensorflow.keras as keras


model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3),include_top=False,weights='imagenet')
model.summary() 
x=GlobalAveragePooling2D()(model.output)
x = keras.layers.Dense(128, activation='relu')(x)
output = keras.layers.Dense(7, activation='sigmoid')(x)
model1 = keras.models.Model(inputs=model.input, outputs=output) 
model1.load_weights('MobileNet_WholeBody_5.h5')
model1.layers[-1].activation = None
model1.summary()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    
def save_and_display_gradcam(idx,prod, img_array, heatmap,ID, alpha=0.4):
    img=img_array*255
    img=img[0,:,:,:]
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    
    heatmap = np.uint8(255 * heatmap)
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img 
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    # Save the superimposed image
    
    ID=ID+"@"+str(prod)
    if idx==0:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/0/'
      superimposed_img.save(path+ID+".jpg")
    if idx==1:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/1/'
      superimposed_img.save(path+ID+".jpg")
    if idx==2:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/2/'
      superimposed_img.save(path+ID+".jpg")
    if idx==3:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/3/'
      superimposed_img.save(path+ID+".jpg")
    if idx==4:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/4/'
      superimposed_img.save(path+ID+".jpg")
    if idx==5:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/5/'
      superimposed_img.save(path+ID+".jpg")
    if idx==6:
      path='/home/user01/data_ssd/Abbas/DeepLearning/CAM/6/'
      superimposed_img.save(path+ID+".jpg")
    
import glob
#from sklearn.utils import shuffle

face_ids=[]
all_Image_ID=[]
for infile in sorted(glob.glob(r'/home/user01/data_ssd/Abbas/DeepLearning/whole body/*.png')):
    face_ids.append(infile)
    all_Image_ID.append(infile[52:-4])
#face_ids, all_Image_ID = shuffle(face_ids,all_Image_ID, random_state=0)

for i in range(len(face_ids)):
  img_array=get_img_array(face_ids[i]) 
  preds = model1.predict(img_array)
  preds=preds.flatten()
  #print(preds)
  sorted_=np.argsort(preds.flatten())[::-1]
  for k, idx in enumerate(sorted_[:1]):
    heatmap = make_gradcam_heatmap(img_array, model1, last_conv_layer_name,pred_index=idx)
    save_and_display_gradcam(idx,preds[idx],img_array, heatmap,all_Image_ID[i])
    #print(idx)
    #print(preds[idx])


  

  
  



