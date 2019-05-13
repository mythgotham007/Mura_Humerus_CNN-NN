#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import tensorflow as tf


# In[2]:


"""Load Train Data"""
#code for reading data
imgdir = []   #shall contain the full path of the image
ytr = []         #shall contain label of each image
#xtr = np.matrix([[]])         #shall contain each image

#xtr = np.zeros((257,262144))   #modified
xtr = np.zeros((1300,10000))

datatype = ['train', 'valid']
studylabel = {'negative':0, 'positive':1}

BASE_DIR = 'murashort'

traindir = BASE_DIR + '/' + datatype[0]  #shows training directory

patients = os.listdir(traindir)  #all the patient ID's are now into this

i = 0
for patient in patients:
    directory = traindir + '/' + patient    #patient directory train/patientxxxx/
    studies = os.listdir(directory)         #creates list of studies in each patient directory
    for study in studies:
        images = os.listdir(directory + '/' + study)
        for image in images:
            imgdir = np.append(imgdir, directory + '/' + study + '/' + image)    #add image addresses in array
            ytr = np.append(ytr, studylabel[study.split('_')[1]])        #keep adding labels of each images
            file = mpimg.imread(imgdir[i])
            if (len(np.shape(file))>2):
                pixels = np.matrix(file[:,:,0])     #takes matrix format of one image
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))   #flattens the image to 1-D
            else:
                pixels = np.matrix(file)
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))
                
            if (np.shape(pixels)[1] < 10000):    #modified from 262144
                zeroarray = np.matrix(np.zeros((1,(10000-np.shape(pixels)[1]))))
                pixels = np.concatenate((pixels,zeroarray), axis=1)
                #print(pixels)
                #xtr = np.append(xtr, np.array(file[:,:,0]), axis=0)
                ###xtr = np.concatenate(xtr,pixels, axis=1)               #load and add images in 
                xtr[i] = pixels               #load and add images in 
                #print ('i from if=', i)
            else:
                xtr[i] = np.matrix(pixels[:,0:10000])     #modified from 262144
                #print(pixels)
                #print ('i from else=', i)
                
            i=i+1
            
xtrain = xtr[0:i,:]
ytrain = ytr
"""Normalize data"""
for i in range (len(xtrain)):
    if np.amax(xtrain[i]) != 0:
        xtrain[i] = xtrain[i] / np.amax(xtrain[i])
"""Eliminating all zero data"""
xtr = np.zeros((np.shape(xtrain)))
ytr = []
j = 0
i = 0
while i < (len(ytrain)):
    if np.amax(xtrain[i]) != 0:
        xtr[j] = xtrain[i]
        ytr.append(ytrain[i])
        j = j+1
    i = i+1
    
xtr = xtr[0:j,:]


# In[3]:


"""Load Test Data"""
#code for reading data
imgdir = []   #shall contain the full path of the image
yts = []         #shall contain label of each image
#xtr = np.matrix([[]])         #shall contain each image

#xtr = np.zeros((257,10000))   #modified
xts = np.zeros((288,10000))

datatype = ['train', 'valid']
studylabel = {'negative':0, 'positive':1}

BASE_DIR = 'murashort'

testdir = BASE_DIR + '/' + datatype[1]  #shows training directory

patients = os.listdir(testdir)  #all the patient ID's are now into this

i = 0
for patient in patients:
    directory = testdir + '/' + patient    #patient directory train/patientxxxx/
    studies = os.listdir(directory)         #creates list of studies in each patient directory
    for study in studies:
        images = os.listdir(directory + '/' + study)
        for image in images:
            imgdir = np.append(imgdir, directory + '/' + study + '/' + image)    #add image addresses in array
            yts = np.append(yts, studylabel[study.split('_')[1]])        #keep adding labels of each images
            file = mpimg.imread(imgdir[i])
            if (len(np.shape(file))>2):
                pixels = np.matrix(file[:,:,0])     #takes matrix format of one image
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))   #flattens the image to 1-D
            else:
                pixels = np.matrix(file)
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))
                
            if (np.shape(pixels)[1] < 10000):    #modified from 262144
                zeroarray = np.matrix(np.zeros((1,(10000-np.shape(pixels)[1]))))
                pixels = np.concatenate((pixels,zeroarray), axis=1)
                #print(pixels)
                #xtr = np.append(xtr, np.array(file[:,:,0]), axis=0)
                ###xtr = np.concatenate(xtr,pixels, axis=1)               #load and add images in 
                xts[i] = pixels               #load and add images in 
                #print ('i from if=', i)
            else:
                xts[i] = np.matrix(pixels[:,0:10000])     #modified from 262144
                #print(pixels)
                #print ('i from else=', i)
                
            i=i+1
            
xtest = xts[0:i,:]
ytest = yts
"""Normalize data"""
for i in range (len(xtrain)):
    if np.amax(xtrain[i]) != 0:
        xtrain[i] = xtrain[i] / np.amax(xtrain[i])
        
        
"""Eliminating all zero data"""
xts = np.zeros((np.shape(xtest)))
yts = []
j = 0
i = 0
while i < (len(ytest)):
    if np.amax(xtest[i]) != 0:
        xts[j] = xtest[i]
        yts.append(ytest[i])
        j = j+1
    i = i+1
    
xts = xts[0:j,:]
"""Verification"""
print(np.shape(xtest))
print(np.shape(xts))
print(np.shape(ytest))
print(np.shape(yts))


# In[4]:


"""Verification"""
print(np.shape(xtrain))
print(np.shape(xtr))
print(np.shape(ytrain))
print(np.shape(ytr))
z=np.zeros((1260,1,100,100))
for i in range(len(xtr)):
    #print(xtr[i].shape)
    z[i,:,:,:]=np.reshape(xtr[i].ravel(),(1,100,100))
    print(z.shape)


# In[5]:


"""Function for calculating error"""         #approved
def errcalc(y,ypred):
    if len(y) != len(ypred):
        print('The inputs must be same size')
        return -1
    else:
        error = np.sum(np.absolute(np.subtract(y,ypred)))
    return error/(len(y))


# In[6]:


"""Function for creating a 5 layer NN. Creates, Compiles and Returns the Model"""   #approved
import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Flatten
def make_cnn( h1size, h2size, h3size):
    #Training data in 'xtr'
    #Labels in 'ytr'

    xxx=Input(shape=(1,100,100))
    h1=Conv2D(h1size, kernel_size=(4, 4), strides=(1, 1),
                 activation='relu',data_format = 'channels_first')(xxx)
    h2=MaxPooling2D(pool_size=(2, 2))(h1)
    h3=Conv2D(h2size, kernel_size=(4, 4),activation='relu')(h2)
    h4=MaxPooling2D(pool_size=(2, 2))(h3)
    h5=Conv2D(h3size,kernel_size=(5, 5), activation='relu')(h4)
    h6=MaxPooling2D(pool_size=(2, 2))(h5)
    h7=Flatten()(h6)
    h8=Dense(h3size*23*23, activation='relu')(h7)
    h9=Dense(500, activation='relu')(h8)
    h10=Dense(50, activation='relu')(h9)
    h11=Dense(5, activation='relu')(h10)
    h12=Dense(1, activation='sigmoid')(h11)
    ###creating the neural network
    shallow_cnn = Model(xxx, h12)
    

    ###Compiling model
    shallow_cnn.compile( loss='binary_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

    ### Returning compiled model
    
    return shallow_cnn


# In[7]:


"""Function for fitting into a model"""        #approved
def train_cnn(shallow_cnn,xtr,ytr, batchsize, epchs, shffle):
    shallow_cnn.fit(xtr, ytr, batch_size=batchsize, epochs=epchs, shuffle=shffle,verbose=1)
    return shallow_cnn


# In[8]:


"""Function for Predicting, and calculating error"""   #approved
def predict_cnn(shallow_cnn, x, y):
    pred = shallow_cnn.predict(x)
    y_pred = np.round(pred)
    y_pred = np.ravel(y_pred)
    valid_err = errcalc(y,y_pred)
    return y_pred, valid_err


# In[9]:


"""Function for calculating Kappa"""
def kappa_calc(y, y_pred):
    
    if len(y) != len(y_pred):
        print('Input array sizes must match')
        return -1
    else:    
        #parameters
        total_data = len(y)
        agree_num = len(np.where(y == y_pred)[0])    #where model and specialists agree

        #specialists saying yes and no
        spec_yes = len(np.where(np.array(y) == 1)[0])
        spec_no = total_data - spec_yes

        #model saying yes
        model_yes = len(np.where(np.array(y_pred) == 1)[0])
        model_no = total_data - model_yes

        #calculating p_o
        p_o = agree_num / total_data

        #calculate p_e [raters randomly agreeing]
        rand_yes = (spec_yes/ total_data) * (model_yes/ total_data)
        rand_no = (spec_no / total_data) * (model_no / total_data)
        p_e = rand_yes + rand_no


        #kappa
        kappa = (p_o - p_e) / (1 - p_e)
        #kappa = 0
        print('kappa value is = ', kappa)
        print('po value is', p_o)
        print('agree num', agree_num)
        print('pe value is', p_e)
        print('rand_yes ', rand_yes)
        print('rand_no ', rand_no)
        print('spec_yes ' , spec_yes)
        print('spec_no', spec_no)
        print('model_yes ', model_yes)
        print('model_no ', model_no)
        return kappa


# In[10]:


"""Function for calculating weighted error Epsilon""" #approved
def weighted_error(y,ypred,d):
    t = len(d)
    vec = np.abs(np.subtract(y,ypred))
    sumerror = 0.0
    for i in range (t):
        sumerror = sumerror + (d[i]*vec[i])
    return sumerror


# In[11]:


def distr_upd(D, epsilon, alpha, y, h_x):       #approved
    z = 2.0* math.sqrt(epsilon*(1-epsilon))
    D_out = np.zeros(len(D))
    ##function to find mismatch vector
    ylabel = np.array(y)
    hypo = np.array(h_x)
    mismatch_vec = np.where(ylabel!=hypo, -1, 1)
    for i in range (len(D)):
        D_out[i] = (D[i]/z) * math.exp( (-alpha)* mismatch_vec[i])
    D_out = D_out / np.sum(D_out)
    return D_out/np.sum(D_out)


# In[12]:


"""Function for Predicting Ensambled output"""   #approved
def ens_pred_nn(shallow_cnn, x, y, alpha):
        nbsample= np.shape(x)[0]
        y_pred = np.zeros((nbsample))
        for nn, alph in zip(shallow_cnn, alpha):
            y_temp, __ = predict_cnn(nn, x, y)
            #added
            y_temp = np.array(y_temp)
            y_temp = np.where(y_temp==0, -1, 1)
            y_pred = y_pred + (alph * y_temp)
        
        y_pred = quantizer(y_pred)
        return y_pred  


# In[13]:


"""Function for Sampling
Takes a 1-by-N matrix and outputs result as a one-dimensional matrix"""   #approved

def sampling(distr, n_samples):
    
    indvec = np.array(range(n_samples))
    out_ind=[]
    
    for i in range (n_samples):
        num = np.ceil(n_samples * distr[i])
        num = int(num)
        for j in range(num):
            out_ind.append(indvec[i])
    np.random.shuffle(out_ind)
    return out_ind[0:n_samples]


# In[14]:


"""Function for quantizing predicted class values by custom AdaBoost implementation   
(so that classes take either 0 or 1 values)"""             #approved
def quantizer(y):
    yout=[]
    for i in range (len(y)):
        if(y[i]<0.0):
            yout.append(0.0)
        else:
            yout.append(1.0)
    yout = np.array(yout)
    return yout




### Adaboost flow
##=================================

# x,y = traindata, trainlabel
x,y = z, ytr
testdata=xts,yts
# initial = D = 1/n
nbsample = len(ytr)
D = (1/nbsample)* np.ones((nbsample,))

# variable to hold neural nets
nn_cluster = []

#variable to hold alpha values
alpha = []


with tf.device('/device:GPU:1'):
    for t in range (10):
        #1. create classifier
        shallow_cnn = make_cnn(256,128,64)

        #2. Train using xtr
        train_cnn(shallow_cnn,x,y, 5, 10, True)

        #3. Predict on xtr
        ypred, __ = predict_cnn(shallow_cnn, x, y)

        #4. calculate epsilon
        epsilon = weighted_error(y,ypred,D)

        #5. Calculate alpha
        alph = 0.5*math.log10((1-epsilon)/epsilon)

        #6. update D
        D = distr_upd(D, epsilon, alph, y, ypred)

        #7. append the already made classifier
        nn_cluster.append(shallow_cnn)

        #8. append the alpha values captured
        alpha.append(alph)

        #9. Sample data for next iteration
        sampled_ind = sampling(D,nbsample)
        x , y = x[sampled_ind,:], np.array(y)[sampled_ind]


# Runs the op.
### By this point, the classifier should be prepared, now we apply the predict function

ytr_predict = ens_pred_nn(nn_cluster, z, ytr, alpha)

### Calculating Error
train_error = errcalc(ytr, ytr_predict)

### Calculating Kappa

train_kappa = kappa_calc(ytr, ytr_predict)


## Printing Results

print('**************==============******************')
print('Training error is ', train_error, ' and kappa is ', train_kappa)


# In[19]:


# Runs the op.
### By this point, the classifier should be prepared, now we apply the predict function

ytr_predict = ens_pred_nn(nn_cluster, z, ytr, alpha)

### Calculating Error
train_error = errcalc(ytr, ytr_predict)

### Calculating Kappa

train_kappa = kappa_calc(ytr, ytr_predict)


## Printing Results

print('**************==============******************')
print('Training error is ', train_error, ' and kappa is ', train_kappa)


# In[ ]:


x


# In[20]:


"""Verification"""
print(np.shape(xts))
print(np.shape(yts))
z1=np.zeros((282,1,100,100))
for i in range(len(xts)):
    #print(xtr[i].shape)
    z1[i,:,:,:]=np.reshape(xts[i].ravel(),(1,100,100))
    print(z1.shape)


# In[21]:


### By this point, the classifier should be prepared, now we apply the predict function

yts_predict = ens_pred_nn(nn_cluster, z1, yts, alpha)

### Calculating Error
test_error = errcalc(yts, yts_predict)

### Calculating Kappa

test_kappa = kappa_calc(yts, yts_predict)


## Printing Results

print('**************==============******************')
print('Testing error is ', test_error, ' and kappa is ', test_kappa)


# In[ ]:


yts.shape


# In[ ]:




