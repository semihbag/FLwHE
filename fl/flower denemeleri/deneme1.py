import flwr as fl
import tensorflow as tf
from typing import Callable, Tuple, Any, List

# Model tanımı
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Girdi boyutu (örnek MNIST için)
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Veri yükleme fonksiyonu
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

# Dinamik Flower istemci sınıfı
class DynamicClient(fl.client.NumPyClient):
    def __init__(self, model_fn: Callable[[], Any], data_fn: Callable[[], Tuple]):
        self.model = model_fn()
        self.x_train, self.y_train, self.x_test, self.y_test = data_fn()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Sunucu ve istemcilerin aynı dosyada çalıştırılması
def main():
    # Veriyi yükle
    data_fn = load_data

    # İstemci sayısı
    num_clients = 3

    # Her istemci için ayrı bir veri dilimi oluştur
    x_train, y_train, x_test, y_test = data_fn()
    client_data_splits = [
        (x_train[i::num_clients], y_train[i::num_clients]) for i in range(num_clients)
    ]

    # İstemcileri oluştur
    clients = [
        DynamicClient(
            model_fn=create_model,
            data_fn=lambda split=split: (*split, x_test, y_test),
        )
        for split in client_data_splits
    ]

    # Flower stratejisini kullanarak sunucuyu başlat
    strategy = fl.server.strategy.FedAvg()

    # Simülasyonu başlat
    fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
