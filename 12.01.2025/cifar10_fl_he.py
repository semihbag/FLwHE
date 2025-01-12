## FL+HE Implementation

# File: fl_he.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time
import tenseal as ts

# Data Preparation
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    return (x_train, y_train), (x_test, y_test)

# Create Model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Homomorphic Encryption Setup
def setup_encryption():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

# Federated Learning with HE Simulation
def federated_learning_with_he(num_clients=5, epochs=5):
    (x_train, y_train), (x_test, y_test) = load_data()
    client_data = []

    context = setup_encryption()

    for i in range(num_clients):
        start_idx = i * len(x_train) // num_clients
        end_idx = (i + 1) * len(x_train) // num_clients
        client_data.append((x_train[start_idx:end_idx], y_train[start_idx:end_idx]))

    # Global model is now encrypted
    global_model = create_cnn_model()
    global_weights = global_model.get_weights()
    encrypted_global_weights = [ts.ckks_vector(context, np.array(w, dtype=np.float64).flatten()) for w in global_weights]

    metrics = []
    start_time = time.time()  # Start time for total elapsed time

    for epoch in range(epochs):
        encrypted_client_weights = []
        client_times = []

        for client_x, client_y in client_data:
            # Client decrypts the global weights for local training
            decrypted_weights = [np.array(w.decrypt()).reshape(global_weights[i].shape) for i, w in enumerate(encrypted_global_weights)]
            client_model = create_cnn_model()
            client_model.set_weights(decrypted_weights)

            start_time_client = time.time()
            client_model.fit(client_x, client_y, epochs=1, verbose=0)
            end_time_client = time.time()

            # Encrypt updated client weights
            client_weights = client_model.get_weights()
            encrypted_weights = [ts.ckks_vector(context, np.array(w, dtype=np.float64).flatten()) for w in client_weights]
            encrypted_client_weights.append(encrypted_weights)
            client_times.append(end_time_client - start_time_client)

        # Aggregate encrypted weights using FedAvg
        aggregated_weights = []
        for i in range(len(encrypted_global_weights)):
            encrypted_sum = encrypted_client_weights[0][i]
            for j in range(1, num_clients):
                encrypted_sum += encrypted_client_weights[j][i]
            aggregated_weights.append(encrypted_sum.mul(1 / num_clients))

        encrypted_global_weights = aggregated_weights

        # Metrics are evaluated on the client side after decrypting the model
        decrypted_global_weights = [np.array(w.decrypt()).reshape(global_weights[i].shape) for i, w in enumerate(encrypted_global_weights)]
        global_model.set_weights(decrypted_global_weights)

        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        avg_time = np.mean(client_times)
        metrics.append((epoch, loss, accuracy, avg_time))
        print(f"Epoch {epoch}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Avg Client Time: {avg_time:.4f}")

    end_time = time.time()  # End time for total elapsed time
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (elapsed_time - int(elapsed_time)) * 1000
    print(f"Total Elapsed Time: {int(hours)}h {int(minutes)}m {int(seconds)}s {int(milliseconds):.0f}ms")

    return metrics

if __name__ == "__main__":
    metrics = federated_learning_with_he()
    print("Metrics:", metrics)