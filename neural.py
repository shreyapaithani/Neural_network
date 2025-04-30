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
w1=np.random.randn(input_size,128)*0.01
b1=np.zeros((1,128))
print("input layer done")

