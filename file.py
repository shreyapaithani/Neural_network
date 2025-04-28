import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from keras.datasets import mnist  # Sirf data load karne ke liye

# ----  MNIST Load & Preprocess ----
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize & Flatten (28x28 → 784)
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# ----  Initialize layerss ----
input_size = 784
hidden_size = 128
output_size = 10


#---initialization of weights

W1 = np.random.randn(input_size, hidden_size) * 0.01 #weight for input layer and first hidden layer
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01 #weight for hidden and output layer
b2 = np.zeros((1, output_size))

# ---- Activation Functions ----
def relu(Z):
    return np.maximum(0, Z) #netwrok learn karane k liye 

def softmax(Z): # digits recognize 
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ---- Forward Propagation ----

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

    loss=compute_loss(Y_test,A2)

# ----- Compute Loss ----
def compute_loss(Y, A2):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / m 
    return loss
# ----- Backpropagation -----
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate=0.01):
    m = X.shape[0]

    # Compute Gradients
    dZ2 = A2 - Y  # Error at output layer
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)  # Backpropagating error
    dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update Weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


# One-hot encode labels
def one_hot_encode(Y, num_classes=10):
    return np.eye(num_classes)[Y]

Y_train_oh = one_hot_encode(Y_train)
Y_test_oh = one_hot_encode(Y_test)

# Training parameters
val = 5
learning_rate = 0.01

# Training Loop
for i in range(val):
    # Forward propagation
    Z1, A1, Z2, A2 = forward_propagation(X_train)

    #---b Compute Loss
    loss = compute_loss(Y_train_oh, A2)

    #---- Backpropagation ----
    W1, b1, W2, b2 = backpropagation(X_train, Y_train_oh, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate)
    print(f"Loss: {loss:.4f}")

print("Training Complete!")

