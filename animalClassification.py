import random
import random
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
img_size=100
input1=r'C:\AnimalClassification\Animals\animals10\raw-img'
categories=["dog","chick","squirel","horse","cat","butterfly"]
training_data=[]
#Taking 8 categories of animals
def create_training_data():
    for category in categories:
            path=os.path.join(input1,category)
            class_num=categories.index(category)
            i=0
            for img in os.listdir(path):
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array=cv2.resize(img_array,(img_size,img_size))
                    training_data.append([new_array,class_num])
           

create_training_data()
#Shuffling the training dataset
random.shuffle(training_data)

X=[]
y=[]
#Taking features and its corresponding labels into 2 lists
for features,label in training_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1,img_size,img_size,1)

animals=(X)
labels=np.array(y)

s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

#Calculating the length of labels
num_classes=len(np.unique(labels))
data_length=len(animals)

#Dividing to train and test variables
(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

train_length=len(x_train)
test_length=len(x_test)


(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

#One hot encoding
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)


# import sequential model and all the required layers

#making the final model : 4 Conv2D layers, 2 MaxPooling layers, Batch normalizations and 5 Drop out layers
cnn = Sequential()
cnn.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu', input_shape = (100,100,1)))
cnn.add(BatchNormalization())

cnn.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(256, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(6, activation = 'softmax'))

#Printing the summary of final model
cnn.summary()


cnn.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
  
cnn.fit(x_train,y_train,batch_size=32,
          epochs=5,verbose=1)

score = cnn.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test accuracy:', score[1])

predicted_classes = cnn.predict_classes(x_test)

Y_True = y_test
Y_True=np.argmax(Y_True, axis=1)
target_names = ['Dog','Chick','Squirrel','Horse','Cat','Butterfly']
print(classification_report(Y_True, predicted_classes, target_names = target_names))
