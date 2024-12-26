import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import tenseal as ts  # Homomorphic Encryption kütüphanesi

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

# Homomorphic Encryption için yardımcı fonksiyonlar
def encrypt_weights(weights, context):
    encrypted_weights = []
    for layer in weights:
        encrypted_weights.append([ts.ckks_vector(context, vec) for vec in layer])
    return encrypted_weights

def decrypt_weights(encrypted_weights, context):
    decrypted_weights = []
    for layer in encrypted_weights:
        decrypted_weights.append([vec.decrypt() for vec in layer])
    return decrypted_weights

def aggregate_encrypted_weights(client_encrypted_weights, context):
    aggregated_weights = []
    for encrypted_layer_tuple in zip(*client_encrypted_weights):
        aggregated_layer = [sum(vecs) / len(vecs) for vecs in zip(*encrypted_layer_tuple)]
        aggregated_weights.append(aggregated_layer)
    return aggregated_weights

# Federated Learning Fonksiyonları
def train_client_model(client_data, epochs=1):
    x, y = client_data
    model = create_model()
    model.fit(x, y, epochs=epochs, verbose=0)
    return model.get_weights()

# Homomorphic Encryption için ayarlar
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2 ** 40
context.generate_galois_keys()

# Homomorphic Encryption ile Federated Learning Eğitimi
server_model = create_model()
num_rounds = 3

for round_num in range(num_rounds):
    print(f"Round {round_num + 1} of Federated Learning with Homomorphic Encryption")
    client_encrypted_weights = []

    for client_data in client_datasets:
        # Client modelini eğit
        client_weight = train_client_model(client_data)

        # Ağırlıkları şifrele
        encrypted_weights = encrypt_weights(client_weight, context)
        client_encrypted_weights.append(encrypted_weights)

    # Sunucuda şifreli ağırlıkları birleştir
    aggregated_encrypted_weights = aggregate_encrypted_weights(client_encrypted_weights, context)

    # Şifreli ağırlıkları çöz
    aggregated_weights = decrypt_weights(aggregated_encrypted_weights, context)

    # Birleştirilmiş ağırlıkları sunucu modeline uygula
    server_model.set_weights(aggregated_weights)

# Modeli test edelim
loss, accuracy = server_model.evaluate(x_test, y_test)
print(f"Federated Learning with Homomorphic Encryption Accuracy: {accuracy * 100:.2f}%")