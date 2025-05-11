epochs = 10         # Number of times the dataset will be used
batch_size = 64     # Number of samples per batch
m = X_train.shape[0]  # Total training samples

for epoch in range(epochs):
    shuffled_indices = np.random.permutation(m)  # Shuffle data for each epoch
    X_train_shuffled = X_train[shuffled_indices]
    Y_train_shuffled = Y_train[shuffled_indices]

    for i in range(0, m, batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        Y_batch = Y_train_shuffled[i:i+batch_size]

        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X_batch)

        # Compute loss
        loss = compute_loss(Y_batch, A2)

        # Backpropagation & weight updates
        backward_propagation(X_batch, Y_batch, Z1, A1, Z2, A2)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("Training Complete!")
