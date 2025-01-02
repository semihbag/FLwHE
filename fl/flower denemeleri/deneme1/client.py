import flwr as fl
from typing import Tuple, Any, Callable

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

if __name__ == "__main__":
    import tensorflow as tf

    # Model tanımı
    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Girdi boyutları
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # Dinamik veri yükleme fonksiyonu
    def load_data():
        # MNIST veri setini yüklemek için örnek
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test

    # İstemciyi başlat
    client = DynamicClient(create_model, load_data)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
