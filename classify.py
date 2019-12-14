import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt


# load labels into memory
labels = []
with open('./data/labels.csv', 'r') as f:
    for line in f.read().split('\n')[1:]:
        labels.append(line.split(',')[-1])

# set path to load model from
model_path = './models/model7.hd5'
# load model
model = tf.keras.models.load_model(model_path)


# we expect a 4D array: (batch size, height, width, channels)
def classify(x):
    # if a single image is given, transform it into 4D array
    if len(np.shape(x)) == 3:
        x = np.array([x])

    # run the network and return
    return model.predict(x)
