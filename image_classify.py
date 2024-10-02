import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import layers 
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

#with open('output.txt', 'w', encoding='utf-8') as f:
#    sys.stdout = f
    # Continue with your code here



data_train_path = './Fruits_Vegetables/train'
data_test_path = './Fruits_Vegetables/train'
data_val_path = './Fruits_Vegetables/validation'

img_width = 180 
img_height = 180 

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size = (img_width,img_height),
    batch_size = 32 ,
    validation_split = False)


print(data_train.class_names)
data_cat = data_train.class_names 

data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle=True,
    image_size = (img_width,img_height),
    batch_size = 32 ,
    validation_split = False)

data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle=True,
    image_size = (img_width,img_height),
    batch_size = 32 ,
    validation_split = False)

plt.figure(figsize=(10,10))
for image ,labels in data_train.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(data_cat[int(labels[i])])
        plt.axis('off')
# plt.show()

from tensorflow.keras.models import Sequential 

data_train

model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics= ['accuracy'])

epochs_size= 25
history = model.fit(data_train,validation_data = data_val,epochs = epochs_size)


# Graph of accuracy and Loss
epochs_range = range(epochs_size)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,history.history['accuracy'],label= 'Training accuracy')
plt.plot(epochs_range,history.history['val_accuracy'],label= 'Validation accuracy')
plt.title('Accuracy')

plt.subplot(1,2,1)
plt.plot(epochs_range,history.history['loss'],label= 'Training Loss')
plt.plot(epochs_range,history.history['val_loss'],label= 'validation Loss')
plt.title('Loss')


image = './Fruits_Vegetables/Image_3.jpg'
image = tf.keras.utils.load_img(image,target_size=(img_height,img_width))
img_arr =tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
print('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))

model.save('Image_classify.keras')