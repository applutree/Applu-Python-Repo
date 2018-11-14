# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:52:37 2017

@author: applu
"""
import os
import skimage
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import random

from skimage import data
from skimage.color import rgb2gray
from skimage import transform 

""" Function Definition """
# Data loading function -------------------------------------------------------
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    #print(directories)
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        #print(label_directory)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        #print(file_names)
        for f in file_names:
            #print(str(f))
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels
# Data loading function -------------------------------------------------------

""" Directory assignments """    
ROOT_PATH = "."
train_data_directory = os.path.join(ROOT_PATH, "datasets", "TrafficSigns", "Training")
test_data_directory = os.path.join(ROOT_PATH, "datasets", "TrafficSigns", "Testing")
images, labels = load_data(train_data_directory)
#------------------------------------------------------------------------------

""" Construction phase of Tensorflow """
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)
# Fully connected layer - Operation that takes care of creating variables that are used internally
# logits is defined as the activation function - in our case we are use tf.nn.relu aka ReLu
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function - Keep track of loss for optimizer to minimize
# In this case we are using softmax cross entropy with logits - it will measure the performance of a classification model
# whose output value is between 1 and 0
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

"""----Define Optimizer here----"""
# Define an optimizer -------------
#train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
#train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)
# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Resize all images to 28 x 28 and assign images28
images28 = [transform.resize(image, (28, 28)) for image in images]
# Convert `images28` to an array and then to grayscale
images28 = rgb2gray(np.array(images28))

""" Execution Phase of Tensorflow """
tf.set_random_seed(42)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# An EPOCH is an iteration of training cycle
for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

""" Model Evaluation """
# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()

# Load the test data
test_images, test_labels = load_data(test_data_directory)
# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]
# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))
# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
# Calculate the accuracy
accuracy = match_count / len(test_labels)
# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()