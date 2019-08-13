import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os, shutil


main_dir = '/Users/Prakash/Downloads/cell_images'  # Folder downloaded to this dir

train_dir = os.path.join(main_dir, 'train')  # Create train dir
os.mkdir(train_dir)

validation_dir = os.path.join(main_dir, 'validation')  # Create validation dir
os.mkdir(validation_dir)

os.chdir('/Users/Prakash/Downloads/cell_images/Parasitized')
i = 1
for fname in os.listdir('/Users/Prakash/Downloads/cell_images/Parasitized'):  # Rename image files
    src = fname
    dst = 'Parasitized' + str(i) + '.png'
    os.rename(src, dst)
    i += 1

os.chdir('/Users/Prakash/Downloads/cell_images/Uninfected')
i = 1
for fname in os.listdir('/Users/Prakash/Downloads/cell_images/Uninfected'):  # Rename image files
    src = fname
    dst = 'Uninfected' + str(i) + '.png'
    os.rename(src, dst)
    i += 1

Parasitized_dir = '/Users/Prakash/Downloads/cell_images/Parasitized'
Uninfected_dir = '/Users/Prakash/Downloads/cell_images/Uninfected'

train_uninfected_dir = os.path.join(train_dir, 'uninfected')  # Create uninfected dir within train dir
os.mkdir(train_uninfected_dir)

train_parasitized_dir = os.path.join(train_dir, 'parasitized')  # Create parasitized dir within train dir
os.mkdir(train_parasitized_dir)

validation_uninfected_dir = os.path.join(validation_dir, 'uninfected')  # Create uninfected dir within validation dir
os.mkdir(validation_uninfected_dir)

validation_parasitized_dir = os.path.join(validation_dir, 'parasitized')  # Create parasitized dir within validation dir
os.mkdir(validation_parasitized_dir)


fnames = ['Parasitized{}.png'.format(i) for i in range(1, 1001)]  # Move first 1000 images to train_parasitized_dir
for fname in fnames:
    src = os.path.join(Parasitized_dir, fname)
    dst = os.path.join(train_parasitized_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['Parasitized{}.png'.format(i) for i in range(1001, 1501)]  # Move next 500 images to validation_parasitized_dir
for fname in fnames:
    src = os.path.join(Parasitized_dir, fname)
    dst = os.path.join(validation_parasitized_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['Uninfected{}.png'.format(i) for i in range(1, 1001)]  # Move first 1000 images to train_uninfected_dir
for fname in fnames:
    src = os.path.join(Uninfected_dir, fname)
    dst = os.path.join(train_uninfected_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['Uninfected{}.png'.format(i) for i in range(1001, 1501)]  # Move next 500 images to validation_uninfected_dir
for fname in fnames:
    src = os.path.join(Uninfected_dir, fname)
    dst = os.path.join(validation_uninfected_dir, fname)
    shutil.copyfile(src, dst)


test_images = []  # Test array of 5 random images for visualisation
test_imgs = ['Parasitized{}.png'.format(i) for i in range(1, 6)]
for img in test_imgs:
    path = os.path.join(train_parasitized_dir, img)
    test_images.append(plt.imread(path))

fig = plt.figure()  # Plot the 5 images of parasitized cells
ax1 = fig.add_subplot(1, 5, 1)
ax1.imshow(test_images[0])
ax2 = fig.add_subplot(1, 5, 2)
ax2.imshow(test_images[1])
ax3 = fig.add_subplot(1, 5, 3)
ax3.imshow(test_images[2])
ax4 = fig.add_subplot(1, 5, 4)
ax4.imshow(test_images[3])
ax5 = fig.add_subplot(1, 5, 5)
ax5.imshow(test_images[4])


# MODEL BUILDING & COMPILING

from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# BUILDING TRAIN & VALIDATION GENERATORS FOR DATA AUGMENTATION

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (64, 64),
    batch_size = 20,
    class_mode = 'binary')

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (64, 64),
    batch_size = 20,
    class_mode = 'binary')

# MODEL FITTING

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 20,
    validation_data = validation_generator,
    validation_steps = 50)

# PLOTTING OF ACCURACY AND LOSS

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 21)

plt.plot(epochs, acc, 'bo', label = 'Training Acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation Acc')
plt.show()  # Plots Training Acc and Validation Acc against Epochs

plt.plot(epochs, loss, 'ro', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.show()  # Plots Training Loss and Validation Loss against Epochs


# Plot shows overfitting after 10 epochs
