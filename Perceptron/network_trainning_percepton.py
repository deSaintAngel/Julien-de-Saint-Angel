# -*- coding: utf-8 -*-
"""
Created on thu oct 23 19:04:28 2018

@author: julien de Saint Angel
"""


#%%

"""
This program create a matrix W witch countain weighs (w1,w2 ...) of each layers of the neuronal network.
W are save in a npy file 
The neuronal network model use is : the perceptron
This program needs the "mnist.npz" file witch countain exemples for train and test      
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

###############################################################################
# loading files exemples
###############################################################################
npzfile = np.load("data_0.npz")
x_train = npzfile['x_train']
x_test  = npzfile['x_test']
y_train = npzfile['y_train']
y_test  = npzfile['y_test']
plt.scatter(x_train[:,0], x_train[:,1], c=y_train[:,1])
    
###############################################################################

#loadinf files  
img_v=x_train
#img_v=np.reshape(images, (70000, 784)) # images to vector
 
labels = y_train

###############################################################################
# parameters : neuronal network
###############################################################################
l = 2                         # images length 784 => 756 => 720(after convolution)

nl1 = 3                       # numbers of neurons  layers 1
nl2 = 4                        # numbers of neurons  layers 2

n_class = 2                    # numbers of neurons  : output = numbers of class

hc_neurons = np.array([nl1,nl2])    # numbers of neurons / layers

alpha = 0.001                   # learning rate [0.001]
ind_img_start = 1               # numbers of exemples use for training [☺1]

steps_in = 5000
0                # nbr iteration by data input ( image) [5000]
steps_data = 1004
                # nbr iteration in input vector [10 000]

###############################################################################    
# weigth vector initialization : random  : normal law : IF W EXIST
def structure_network(hc_n):
    Wij=[]
    
    for k in range(0,len(hc_n)+1):
        file_name = "W_up" + str(k)+ ".npy"
        print(file_name)
        Wij.append( np.load(file_name))
    return Wij
    
# weigth vector initialization : random  : normal law / IF W DOESN'T EXIST
def structure_network0(l,hc_n,n_class):
    Wij=[]
    W_0 = np.random.normal(0,0.001,l*hc_n[0]) #  random vector size lxhc_n
    Wij.append(np.reshape(W_0,(l,hc_n[0]))) # vector to matrix transformation
    
    ind=1
    for i in np.arange(1,np.size(hc_n)):
        w = np.random.normal(0,0.001,hc_n[i-1]*hc_n[i]) 
        Wij.append(np.reshape(w,(hc_n[i-1],hc_n[i])))
        ind +=1
    
    w_f = np.random.normal(0,0.001,hc_n[ind-1]*n_class) 
    Wij.append(np.reshape(w_f,(hc_n[ind-1],n_class)))
    
    return Wij

W = structure_network0(l,hc_neurons,n_class)

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
def weigth_update(Wij,alpha,A,Deltat,j):
    Wij = Wij + alpha * np.outer(A,Deltat)# A D^t = A transpose deltat : matrix number of ligne = number of neurons and numbers of cowlon = number of grad   
    return Wij


## int indice to vector to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)
def to_one_hot(label_max, k):
    one_hot = np.zeros(k)
    one_hot[label_max] = 1
    return one_hot  

  
###############################################################################    
############################ trainning neuronal network #######################
###############################################################################

## propagation :
def propagation_layer(X_input, Wij):
    Y_output = Activity(X_input,Wij)  
    Y_output = sigmoid(Y_output)
    return Y_output

## trainning loop 
def trainning(X_ini,W_i,Label,alpha,N_class):
    ### neurons activation by layers
    Activ_matrix = []
    Activ_matrix.append(X_ini)
    for i in np.arange(len(W_i)):
        Activ_matrix.append(propagation_layer(Activ_matrix[-1],W_i[i]))

    ### retropropagation
    ##deltat_initial
    Deltat_m = []
    #output_vector = to_one_hot(Label, N_class)
    output_vector = Label
    Deltat_m.insert(0,lost(Activ_matrix[-1], output_vector))
    ##deltat iterations
    for k in np.arange(len(W_i)-1,0,-1):
        sigm_prim = sigmoid_prime(np.array([Activ_matrix[k]]))
        Deltat_m.insert(0,grad_update(W_i[k],Deltat_m[0],sigm_prim))   

    ## weigth update
    W_up=[]
    for j in np.arange(len(W_i)):

        W_up.append(weigth_update(W_i[j],alpha,Activ_matrix[j],Deltat_m[j],j))
        
    return W_up,Activ_matrix,Deltat_m


def prediction(X_ini,W_i):
    ### neurons activation by layers
    Activ_matrix = []
    Activ_matrix.append(X_ini)
    for i in np.arange(len(W_i)):
        Activ_matrix.append(propagation_layer(Activ_matrix[-1],W_i[i]))
    return Activ_matrix

#%%
###############################################################################    
###################### trainning neuronal process #############################
###############################################################################
# patch_vector : input vector coresponding to first neurons activities
X_input = img_v[ind_img_start-1,:]

Label_data = labels[ind_img_start-1]

W_u,Activ0,Deltat0 = trainning(X_input,W,Label_data,alpha,n_class)

for i in range(len(W)):
    nom_fichier = "W_up"+str(i)+"ini.npy"
    np.save(nom_fichier, W_u[i])

for s in tqdm(range(steps_data)):
    Label_data = labels[ind_img_start+s]
    X = img_v[ind_img_start+s,:]
    for ss in range(steps_in):
        W_u,Activm,Deltatm = trainning(X,W_u,Label_data,alpha,n_class) 
    #print(np.max(W_u[0]-W[0]))

for i in range(len(W)):
    nom_fichier = "W_up"+str(i)+".npy"
    np.save(nom_fichier, W_u[i])
    

#%%
##Evaluation prediction test
erreur=0
it = np.shape(x_test)[0]

for k in range(it):
    X = x_test[k]
    Label_data = y_test[k]
    A = prediction(X, W)[-1]
    ###
    #aff_vect(X)
    print("la classe est : ",np.argmax(A),Label_data)    
    if np.argmax(A) != np.argmax(Label_data):
        erreur = erreur+ 1
        
reussit =100-((erreur)*(100/it))

print ('le taux de succes des predictions est de ', reussit, '%' )
