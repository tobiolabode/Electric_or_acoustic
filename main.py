import tensorflow as tf
from tensorflow import keras
import cv2
from sklearn.model_selection import train_test_split
from imutils import paths
import os
import random
import numpy as np

EPOCHS = 60
INIT_LR = 1e-3
BS = 32

inputShape = (28, 28, 3)

model = keras.Sequential()    # first set of CONV => RELU => POOL layers

model.add(keras.layers.Conv2D(20, (5, 5), padding="same",
                              input_shape=inputShape))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# first set of CONV => RELU => POOL layers
model.add(keras.layers.Conv2D(20, (5, 5), padding="same",
                              input_shape=inputShape))
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# fully connected
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500))
model.add(keras.layers.Activation('relu'))

# softmax/ classifer
model.add(keras.layers.Dense(2))
model.add(keras.layers.Activation('softmax'))

imagePaths = ("Data/")
filesinA = os.listdir("Data/acoustic_guitar")
filesinE = os.listdir("Data/electric_guitar")
imagePath2 = ("Data/acoustic_guitar/")
imagePath3 = ("Data/electric_guitar/")

folder = []
folder2 = []
for filename1 in filesinE:
    path1 = (imagePath3 + filename1)
    folder2.append(path1)

for filename1 in filesinA:
    path1 = (imagePath2 + filename1)
    folder.append(path1)


# print(folder2)
# print(folder)
# AG = os.join.path(imagePaths,acoustic guitar,)

print("[INFO] loading images...")
data = []
labels = []


# grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images("")))
# random.seed(42)
# random.shuffle(imagePaths)


# loop over the input images
for imagePath2 in folder:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath2)
    image = cv2.resize(image, (28, 28))
    #image = image[..., ::-1].astype(np.float32)
    image = keras.preprocessing.image.img_to_array(image)
    data.append(image)

for imagePath2 in folder2:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath2)
    image = cv2.resize(image, (28, 28))
    #image = image[..., ::-1].astype(np.float32)
    image = keras.preprocessing.image.img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
for file in folder:
    label = 1
    labels.append(label)

for file in folder2:
    label = 0
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(labels.shape)
# print(data)
print(data.shape)

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

trainY = keras.utils.to_categorical(trainY, num_classes=2)
testY = keras.utils.to_categorical(testY, num_classes=2)


aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                                   height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                                   horizontal_flip=True, fill_mode="nearest")


print("[INFO] compiling model...")
opt = keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save("my_model2.h5")
