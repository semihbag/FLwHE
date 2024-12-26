import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

# MNIST datasetini yükleyelim
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Client'ları simüle etmek için veriyi bölelim
num_clients = 3
client_datasets = []
client_size = len(x_train) // num_clients
for i in range(num_clients):
    start = i * client_size
    end = (i + 1) * client_size
    client_datasets.append((x_train[start:end], y_train[start:end]))

# Her bir client için basit bir model tanımlayalım
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Federated Learning fonksiyonları
def train_client_model(client_data, epochs=1):
    x, y = client_data
    model = create_model()
    model.fit(x, y, epochs=epochs, verbose=0)
    return model.get_weights()

def aggregate_weights(client_weights):
    average_weights = []
    for weights_list_tuple in zip(*client_weights):
        average_weights.append(
            np.mean(weights_list_tuple, axis=0)
        )
    return average_weights

# Federated Learning Eğitimi
server_model = create_model()
num_rounds = 3

for round_num in range(num_rounds):
    print(f"Round {round_num + 1} of Federated Learning")
    client_weights = []

    for client_data in client_datasets:
        client_weight = train_client_model(client_data)
        client_weights.append(client_weight)

    aggregated_weights = aggregate_weights(client_weights)
    server_model.set_weights(aggregated_weights)

# Modeli test edelim
loss, accuracy = server_model.evaluate(x_test, y_test)
print(f"Federated Learning Accuracy: {accuracy * 100:.2f}%")
