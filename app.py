import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import  load_model
import streamlit as st 
import numpy as np 
import sys
sys.stdout.reconfigure(encoding='utf-8')



st.header('Image Classification Model ')
model = load_model('./Image_classify.keras')
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

image_height= 180
image_width = 180
image = st.text_input('Enter Image Name','Image.jpg')

image_load = tf.keras.utils.load_img(image,target_size=(image_height,image_width))
img_arr =tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)  
st.image(image,width = 200)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write('with accuracy of ' + str(np.max(score)*100))



