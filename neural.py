import os
os.environment['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
import numpy as np 
from tf.keras.datasets import mnist 
(X_train,Y_train),(X_test,Y_test)=minist.load_data()
X_train=X_train.reshape(X_train.shape[0],-1)/255.0
X_test=X_test.reshape(X_test[0],-1)/255.0
print("Training :{X_train.shape}")
print("Testing:{X_test.shape})
input_size=784
hidden_size=128;
output_size=10;
num_hiddenlayer=16
np.random.seed(42)
parameters['w1']=np.random.randn(input_size,hidden_size)*0.01
parameters['b1']=np.zeros((1,hidden_size))
for l in range(2,num_hidden_layers+1):
parameters[f"w{num_hidden_layers+1}"]=np.random.randn(hidden_size,output_size)*0.01

