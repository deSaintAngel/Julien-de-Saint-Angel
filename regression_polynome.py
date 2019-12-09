# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:35:16 2018

@author: Julien de Saint Angel
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
#%%
x =  np.linspace(-np.pi/2,np.pi/2,100)
y = 0.5*np.cos(x)+0.5

#%%
# Training Data# Train 
train_X = x
train_Y = y

train_x = []
train_y = []

for i in range(len(train_X)):
    train_x.append([train_X[i]])
    train_y.append([train_Y[i]])

#indexs = np.argsort(train_y,axis=0)
#train_x = train_x[indexs]    

#train_X = np.array(train_x)
train_y = np.array(train_y)
# Parameters
lr = 0.001
num_steps = 10000

#%%
tf_targets = tf.placeholder(tf.float32,(None,1),name='tf_targets')
tf_x = tf.placeholder(tf.float32,(None),name='tf_features')

# Weight and Bias# Weigh 
A = tf.Variable(tf.truncated_normal(([1]),stddev=0.1))
B = tf.Variable(tf.truncated_normal(([1]),stddev=0.1))
C = tf.Variable(tf.truncated_normal(([1]),stddev=0.1))
D = tf.Variable(tf.truncated_normal(([1]),stddev=0.1))
E = tf.Variable(tf.truncated_normal(([1]),stddev=0.1))
# Linear regression (Wx + b)
polynome = tf.add(tf.multiply(A,tf.pow(tf_x,4)),tf.multiply(B,tf.pow(tf_x,3)))
polynome = tf.add(polynome,tf.multiply(C,tf.pow(tf_x,2)))
polynome = tf.add(polynome,tf.multiply(D,tf_x))
polynome = tf.add(polynome,E)

#%%
cost = tf.reduce_mean(tf.square(tf.subtract(polynome, tf_targets)))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
#%%
sess =  tf.Session()
sess.run(tf.global_variables_initializer())
# Training
for step in tqdm(range(num_steps)):
        train_data = {tf_x: train_x, tf_targets: train_y}
        a, b, c, d, e, predict = sess.run([A, B, C, D, E, optimizer], feed_dict=train_data)
    

# Graphic display
plt.plot(train_x, train_y, 'ro', label='Original data')
abscisse = np.linspace(np.min(train_x),np.max(train_x),1000)
pts = np.array(a*np.power(abscisse,4) + b*np.power(abscisse,3) + c*np.power(abscisse,2) + d*abscisse + e)
plt.plot(abscisse, pts, label='aproximation')
plt.legend()
plt.show()
val = np.array(a*np.power(train_x,4) + b*np.power(train_x,3) + c*np.power(train_x,2) + d*train_x + e)
print("err :", np.max(train_y-val))
