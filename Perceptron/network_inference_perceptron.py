# -*- coding: utf-8 -*-
"""
Created on thu oct 23 19:04:28 2018

@author: julien de Saint Angel
"""

#%%

"""
This program takes the matrix's weight. He can evaluate the inference for pictures
This program needs the "mnist.npy" file
After inference of exemples, the program compute the accuracy and makes the matrix confusion
Some weigths witch were already trainning can be use ( you must copy and past weigth in
the same folder where they is the program)     
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
##from numpy import linalg as LA
from tqdm import tqdm


#function for display letter ##################################################
def aff_vect(vecteur):

    plt.imshow(vecteur.reshape((28,28)),cmap=cm.Greys_r)
    plt.show()
    
###############################################################################

###############################################################################

#loadinf files  
data = np.load("mnist.npz")
data.files
images=data["images"]
img_v=np.reshape(images, (70000, 784)) # images to vector
 
labels = data["labels"]

###############################################################################
# parameters : neuronal network
###############################################################################
l = len(img_v[0])     # images length 784
hide_layer_nbr = 2 # numbers of hide layers
nl1 = 120    # numbers of neurons  layers 1
nl2 = 100    # numbers of neurons  layers 2

n_class = 10    # numbers of neurons  : output = numbers of class

hc_neurons = np.array([nl1,nl2]) # numbers of neurons / layers

alpha = 0.0001 # learning rate
#ind_img_start = 1 # numbers of exemples use for training
steps_in = 10 # nbr iteration by data input ( image)
steps_data = 1 # nbr iteration in input vector

############################################################################### 
W = []
W.append( np.load("W_up0.npy"))
W.append(np.load("W_up1.npy"))
W.append(np.load("W_up2.npy"))

###############################################################################    
############################ function for network #############################
###############################################################################

#neuronal activities i+1 according to neurons i
def Activity(neuron_i,weigth_i):
    A = neuron_i @ weigth_i
    #A = (neuron_i @ weigth_i) + biais
    return A

#activation function [0,1] ####################################################
## sigmoid function
def sigmoid(z):
    s = 1.0/(1.0+np.exp(-z))
    return s

## function of retro-propagation ##############################################
## dDerivative of the sigmoid function
def sigmoid_prime(z):
    D_s = z*(1-z) #sigmoid(z)*(1-sigmoid(z))
    return D_s


## lost_function
def lost(activity, out_val):
    c = out_val - activity
    return c

## grad_function :  estimation deltat(layers n-1) from deltat(grad n)
def grad_update(Wij,deltat_k,sigm_prim):
    deltat_j = Wij@deltat_k.T # each lines countains neuronal retro activity : sum of Wij*aj+1
    deltat = sigm_prim * deltat_j # sigmoide(aj-1)(1-sigmoid(aj-1)) * sum of Wij*aj
    return deltat[0]

## weigth update function
def weigth_update(Wij,alpha,A,Deltat):
    Wij = Wij + alpha * np.outer(A,Deltat)# A D^t = A transpose deltat : matrix number of ligne = number of neurons and numbers of cowlon = number of grad   
    return Wij


## int indice to vector to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)
def to_one_hot(label_max, k):
    one_hot = np.zeros(k)
    ind_max = np.argmax(label_max)
    one_hot[ind_max] = 1
    return one_hot 


###############################################################################    
############################ prediction neuronal network #######################
###############################################################################

## propagation :
def propagation_layer(X_input, Wij):
    Y_output = Activity(X_input,Wij)  
    Y_output = sigmoid(Y_output)
    return Y_output

def prediction(X_ini,W_i):
    ### neurons activation by layers
    Activ_matrix = []
    Activ_matrix.append(X_ini)
    for i in np.arange(len(W_i)):
        Activ_matrix.append(propagation_layer(Activ_matrix[-1],W_i[i]))
    return Activ_matrix
        
# patch_vector : input vector coresponding to first neurons activities
#########################################################################################################        
erreur=0
it = 5000
mat_conf = np.zeros((10,10))
for k in range(it):
    ind_img_start = k-9999
    ###
    X = img_v[ind_img_start-1,:]
    Label_data = labels[ind_img_start-1]
    A = prediction(X, W)[-1]
    ###
    #aff_vect(X)
    print("le chiffre est : ",np.argmax(A),Label_data)
    plt.show()
    
    if np.argmax(A) != Label_data:
        erreur = erreur+ 1
    
    mat_conf[Label_data,np.argmax(A)] += 1

reussit =100-((erreur)*(100/it))

print ('le taux de succes des predictions est de ', reussit, '%' )    

plt.imshow(mat_conf,cmap=cm.Greys_r)    