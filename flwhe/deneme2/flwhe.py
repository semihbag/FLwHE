import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import tenseal as ts  # Homomorphic Encryption kütüphanesi
import random

# Rastgelelikleri sabitlemek
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


# # MNIST datasetini yükleyelim
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# CIFAR-100 datasetini yükleyelim
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Client'ları simüle etmek için veriyi bölelim
num_clients = 3
client_datasets = []
client_size = len(x_train) // num_clients
for i in range(num_clients):
    start = i * client_size
    end = (i + 1) * client_size
    client_datasets.append((x_train[start:end], y_train[start:end]))

# # Her bir client için basit bir model tanımlayalım
# def create_model():
#     model = Sequential([
#         Flatten(input_shape=(28, 28)),
#         Dense(128, activation='relu'),
#         Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

def create_model():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),  # CIFAR-100 giriş boyutu
        Dense(128, activation='relu'),
        Dense(100, activation='softmax')  # CIFAR-100, 100 sınıfa sahiptir
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



# Homomorphic Encryption için yardımcı fonksiyonlar
def encrypt_weights(weights, context):
    encrypted_weights = []
    shapes = []  # Şekilleri kaydet
    for layer in weights:
        if isinstance(layer, np.ndarray):
            shapes.append(layer.shape)  # Şekli kaydet
            layer_flattened = layer.flatten()
            encrypted_layer = ts.ckks_vector(context, layer_flattened.tolist())
            encrypted_weights.append(encrypted_layer)
    return encrypted_weights, shapes

def decrypt_weights(encrypted_weights, shapes, context):
    decrypted_weights = []
    for encrypted_layer, shape in zip(encrypted_weights, shapes):
        decrypted_layer = np.array(encrypted_layer.decrypt())  # Şifre çöz
        decrypted_weights.append(decrypted_layer.reshape(shape))  # Şekli geri yükle
    return decrypted_weights

def aggregate_encrypted_weights(client_encrypted_weights, context):
    aggregated_weights = []
    for encrypted_layer_tuple in zip(*client_encrypted_weights):
        aggregated_vector = encrypted_layer_tuple[0].copy()
        for vec in encrypted_layer_tuple[1:]:
            aggregated_vector += vec
        aggregated_vector *= (1 / len(encrypted_layer_tuple))
        aggregated_weights.append(aggregated_vector)
    return aggregated_weights

# Federated Learning Fonksiyonları
def train_client_model(client_data, epochs=1):
    x, y = client_data
    model = create_model()
    model.fit(x, y, epochs=epochs, verbose=0, shuffle=False)
    return model.get_weights()

# Homomorphic Encryption için ayarlar
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2 ** 40
context.generate_galois_keys()

# Homomorphic Encryption ile Federated Learning Eğitimi
server_model = create_model()
num_rounds = 2

for round_num in range(num_rounds):
    print(f"Round {round_num + 1} of Federated Learning with Homomorphic Encryption")
    client_encrypted_weights = []
    client_shapes = []

    for client_data in client_datasets:
        # Client modelini eğit
        client_weights = train_client_model(client_data)

        # Ağırlıkları şifrele
        encrypted_weights, shapes = encrypt_weights(client_weights, context)
        client_encrypted_weights.append(encrypted_weights)
        client_shapes = shapes  # Şekilleri kaydet (tüm client'lar için aynı)

    # Sunucuda şifreli ağırlıkları birleştir
    aggregated_encrypted_weights = aggregate_encrypted_weights(client_encrypted_weights, context)

    # Şifreli ağırlıkları çöz
    aggregated_weights = decrypt_weights(aggregated_encrypted_weights, client_shapes, context)

    # Birleştirilmiş ağırlıkları sunucu modeline uygula
    server_model.set_weights(aggregated_weights)

# Modeli test edelim
loss, accuracy = server_model.evaluate(x_test, y_test, verbose=0)
print(f"Federated Learning with Homomorphic Encryption Accuracy: {accuracy * 100:.2f}%")