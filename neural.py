import os
os.environment['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
import numpy as np 
from tf.keras.datasets import mnist 
(X_train,Y_train),(X_test,Y_test)=minist.load_data()
