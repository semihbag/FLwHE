# File: fl_he.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time
import tenseal as ts

# Data Preparation
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return (x_train, y_train), (x_test, y_test)

# Create Model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Homomorphic Encryption Setup
def setup_encryption():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
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

    global_model = create_cnn_model()
    global_weights = global_model.get_weights()

    metrics = []
    
    for epoch in range(epochs):
        encrypted_client_weights = []
        client_times = []

        for client_x, client_y in client_data:
            client_model = create_cnn_model()
            client_model.set_weights(global_weights)
            client_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            start_time = time.time()
            client_model.fit(client_x, client_y, epochs=1, verbose=0)
            end_time = time.time()

            # Encrypt client weights
            client_weights = client_model.get_weights()
            encrypted_weights = [ts.ckks_vector(context, np.array(w, dtype=np.float64).flatten()) for w in client_weights]
            encrypted_client_weights.append(encrypted_weights)
            client_times.append(end_time - start_time)

        # Aggregate encrypted weights using FedAvg
        aggregated_weights = []
        for i in range(len(global_weights)):
            encrypted_sum = encrypted_client_weights[0][i]
            for j in range(1, num_clients):
                encrypted_sum += encrypted_client_weights[j][i]
            encrypted_sum = encrypted_sum.mul(1 / num_clients)  # Scale by dividing each element
            decrypted_weights = encrypted_sum.decrypt()  # Decrypt the aggregated weights
            aggregated_weights.append(np.array(decrypted_weights).reshape(global_weights[i].shape))

        global_weights = aggregated_weights
        global_model.set_weights(global_weights)

        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        avg_time = np.mean(client_times)
        metrics.append((epoch, loss, accuracy, avg_time))
        print(f"Epoch {epoch}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Avg Client Time: {avg_time:.4f}")

    return metrics

if __name__ == "__main__":
    metrics = federated_learning_with_he()
    print("Metrics:", metrics)
