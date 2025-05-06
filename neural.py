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
parameters={}
#input layer to first hidden layer
parameters['w1']=np.random.randn(input_size,hidden_size)*0.01
parameters['b1']=np.zeros((1,hidden_size))
#hidden layers 
for l in range(2,num_hiddenlayer+1):
parameters[f"w{num_hiddenlayer+1}"]=np.random.randn(hidden_size,output_size)*0.01
parameters[f"b{1}"]=np.zeros((1,hidden_size))
# last hidden layer to output layer
parameters[f"W{num_hiddenlayer + 1}"] = np.random.randn(hidden_size, output_size) * 0.01
parameters[f"b{num_hiddenlayer + 1}"] = np.zeros((1, output_size))
print("16 hidden layer initialised")
#activation function
def relu(Z):
return np.maximum (0,Z)
def softmax(Z):
exp_Z=np.exp(Z-np.max(Z,axis=1,keepdims=True))
return exp_Z/np.sum(exp_Z,axis=1,keepdims=True)
#forward propagation 
def forward_propagation(X,parameters):
A=X
cache={'A0':A}
#process all hidden layers 
for l in range (1,num_hiddenlayer+1):
Z=np.dot(A,parameters[f"W{1}"])+parameters[f"b{1}]
A=relu(Z)
cache[f"Z{1}"]=Z
cache[f"A{1}"]=A
#output layer 
Z_out=np.dot(A,parameters[f"W{num_hiddenlayer +1}"]
A_out=softmax(Z_out)
cache[f"Z{num_hidden_layers + 1}"] = Z_out
cache[f"A{num_hidden_layers + 1}"] = A_out
return cache 
#forward propagation running
cache=forward_propagation(X_test,parameters)
 print("forward propagation with 16 layer completed ")



