import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np #To do array operations.
import os # To iterate through directories and join paths.
import cv2 # To do image operations.
import pickle

# The dataset consists of 30X30 images of Uninfected and Parasitized cells.
training_data = []
IMG_SIZE = 30

def create_training_data(DATADIR,CATEGORIES):
    for category in CATEGORIES:  # for clases "Uninfected" and "Parasitized".
        path = os.path.join(DATADIR,category)  # Create path to "Uninfected" and "Parasitized" classes individually.
        class_num = CATEGORIES.index(category)  # Get the classification  (0 or a 1). "0 - Uninfected" and "1 - Parasitized".
        for img in tqdm(os.listdir(path)):  # Iterate over each image per "Uninfected" and "Parasitized"
            try:
                img_array = cv2.imread(os.path.join(path,img))  # Convert image to array.
                #Displays the image.
                #plt.imshow(img_array)
                #plt.show()
                #print(img_array)
                #print(img_array.shape)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size.
                training_data.append([new_array, class_num])  # add this to our training_data.
            except Exception as e:  # in the interest in keeping the output clean...
                pass
               #except OSError as e:
                #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                #except Exception as e:
                 #   print("general exception", e, os.path.join(path,img))
    return training_data

def convert(training_data):
    #X and Y are now in array form.
    X = []
    y = []
    for features,label in training_data:
            X.append(features)
            y.append(label)
    print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    # Convert into numpy
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
    return X , y