from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import os
import math
import numpy
from PIL import Image

batch_size = 128
num_classes = 1000
epochs = 10
data_augmentation = False

label_counter = 0

training_images = []
training_labels = []

for subdir, dirs, files in os.walk('/data/datasets/imagenet_resized/train/'):
    for folder in dirs:
        for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
            for file in folder_files:
                training_images.append(os.path.join(folder_subdir, file))
                training_labels.append(label_counter)

        label_counter = label_counter + 1

nice_n = math.floor(len(training_images) / batch_size) * batch_size

print(nice_n)
print(len(training_images))
print(len(training_labels))

import random

perm = list(range(len(training_images)))
random.shuffle(perm)
training_images = [training_images[index] for index in perm]
training_labels = [training_labels[index] for index in perm]

def get_batch():
    index = 1

    global current_index

    B = numpy.zeros(shape=(batch_size, 256, 256, 3))
    L = numpy.zeros(shape=(batch_size))

    while index < batch_size:
        img = load_img(training_images[current_index])
        B[index] = img_to_array(img)
        B[index] /= 255

        L[index] = training_labels[current_index]

        index = index + 1
        current_index = current_index + 1

    return B, keras.utils.to_categorical(L, num_classes)

model = Sequential()

# conv1
model.add(Conv2D(16, (3, 3), padding='same', input_shape= (256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv2
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv3
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv4
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv5
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# fc1
model.add(Dense(2048))
model.add(Activation('relu'))

# fc2
model.add(Dense(num_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

for i in range(0, epochs):
    current_index = 0

    while current_index + batch_size < len(training_images):
        b, l = get_batch()

        loss, accuracy = model.train_on_batch(b, l)
        print('batch {}/{} loss: {} accuracy: {}'.format(int(current_index / batch_size), int(nice_n / batch_size), loss, accuracy))

current_index = 0

while current_index + batch_size < len(training_images):
    b, l = get_batch()

    score = model.evaluate_on_batch(b, l, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
