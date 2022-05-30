import matplotlib.pyplot as plt
import seaborn as sns

import keras 
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
import keras_metrics

from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
labels = []

img_size = 224
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data,dtype=object)

dir_path = r'.\Input\archive\Train\**'
for file in glob.glob(dir_path):
    labels.append(file[22:])
    
train = get_data("./Input/archive/Train")
val = get_data("./Input/archive/Test")

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


# DL CNN model with 3 Convolutional layers followed by max-pooling layers
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(1024,activation=('relu'),input_dim=50176))
model.add(Dense(512,activation=('relu'))) 
model.add(Dense(256,activation=('relu'))) 
model.add(Dense(128,activation=('relu')))
model.add(Dense(44,activation=('softmax'))) 
model.summary()
#SVM
model_feat = Model(inputs=model.input,outputs=model.get_layer('dense').output)
feat_train = model_feat.predict(x_train)
feat_test = model_feat.predict(x_val)

svm = SVC(kernel='rbf')
svm.fit(feat_train,y_train)

print(svm.score(feat_test,y_val))

# # De tree
model_feat = Model(inputs=model.input,outputs=model.get_layer('dense').output)
feat_train = model_feat.predict(x_train)
feat_test = model_feat.predict(x_val)
dt = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=20, min_samples_leaf=50)
dt.fit(feat_train,y_train)

print(dt.score(feat_test,y_val))

# # #DL Convolutional
opt = Adam(learning_rate=0.01)
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = opt, metrics = ['accuracy',keras_metrics.precision(), keras_metrics.recall()])

history1 = model.fit(x_train,y_train,steps_per_epoch = int(len(x_train)/20),epochs=20, validation_data = (x_val, y_val))

predictions = model.predict(x_val).argmax(axis=1)

#Plotting the training and validation loss and accuracy
f,ax=plt.subplots(2,1) 

#Loss
ax[0].plot(history1.history['loss'],color='b',label='Training Loss')
ax[0].plot(history1.history['val_loss'],color='r',label='Validation Loss')

#Accuracy
ax[1].plot(history1.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(history1.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.show()
while True:
    pass