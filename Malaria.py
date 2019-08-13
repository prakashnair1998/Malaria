import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os, shutil


main_dir = '/Users/Prakash/Downloads/cell_images'
train_dir = os.path.join(main_dir,'train')
os.mkdir(train_dir)

validation_dir = os.path.join(main_dir, 'validation')
os.mkdir(validation_dir)

os.chdir('/Users/Prakash/Downloads/cell_images/Parasitized')
i = 1
for fname in os.listdir('/Users/Prakash/Downloads/cell_images/Parasitized'):
    src = fname
    dst = 'Parasitized' + str(i) + '.png'
    os.rename(src,dst)
    i+=1

os.chdir('/Users/Prakash/Downloads/cell_images/Uninfected')
i = 1
for fname in os.listdir('/Users/Prakash/Downloads/cell_images/Uninfected'):
    src = fname
    dst = 'Uninfected' + str(i) + '.png'
    os.rename(src, dst)
    i+=1

Parasitized_dir = '/Users/Prakash/Downloads/cell_images/Parasitized'
Uninfected_dir = '/Users/Prakash/Downloads/cell_images/Uninfected'

train_uninfected_dir = os.path.join(train_dir, 'uninfected')
os.mkdir(train_uninfected_dir)

train_parasitized_dir = os.path.join(train_dir, 'parasitized')
os.mkdir(train_parasitized_dir)

validation_uninfected_dir = os.path.join(validation_dir, 'uninfected')
os.mkdir(validation_uninfected_dir)

validation_parasitized_dir = os.path.join(validation_dir, 'parasitized')
os.mkdir(validation_parasitized_dir)


fnames = ['Parasitized{}.png'.format(i) for i in range(1,1001)]
for fname in fnames:
    src = os.path.join(Parasitized_dir, fname)
    dst = os.path.join(train_parasitized_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['Parasitized{}.png'.format(i) for i in range(1001,1501)]
for fname in fnames:
    src = os.path.join(Parasitized_dir, fname)
    dst = os.path.join(validation_parasitized_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['Uninfected{}.png'.format(i) for i in range(1,1001)]
for fname in fnames:
    src = os.path.join(Uninfected_dir, fname)
    dst = os.path.join(train_uninfected_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['Uninfected{}.png'.format(i) for i in range(1001,1501)]
for fname in fnames:
    src = os.path.join(Uninfected_dir, fname)
    dst = os.path.join(validation_uninfected_dir, fname)
    shutil.copyfile(src, dst)



