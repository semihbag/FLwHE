import tensorflow as tf
import tensorflow_federated as tff

# Veri kümesini hazırlayın (MNIST)
def preprocess(dataset):
    return (dataset
            .map(lambda x: (tf.reshape(x['pixels'], [-1]), tf.cast(x['label'], tf.int32)))
            .shuffle(buffer_size=1000)
            .batch(20))

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

mnist_train, mnist_test = tff.simulation.datasets.emnist.load_data()

client_ids = mnist_train.client_ids[:10]  # 10 istemci seçiyoruz.
federated_train_data = make_federated_data(mnist_train, client_ids)

# Model tanımı
def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
    ])

def model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_keras_model(),
        input_spec=preprocess(mnist_train.create_tf_dataset_for_client(client_ids[0])).element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated learning algoritmasını oluşturun
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn)

# Eğitimi başlatın
state = iterative_process.initialize()

for round_num in range(1, 11):  # 10 eğitim turu
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f'Round {round_num}, Metrics={metrics}')

# Eğitimden sonra merkezi test
final_model = state.model
keras_model = create_keras_model()
keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Eğitilmiş ağırlıkları Keras modeline yükleyin
tff.learning.assign_weights_to_keras_model(final_model, keras_model)

# Test setinde değerlendirme
test_data = preprocess(mnist_test.create_tf_dataset_from_all_clients())
results = keras_model.evaluate(test_data, verbose=0)
print(f'Test Metrics: {results}')
