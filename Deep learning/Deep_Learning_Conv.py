import tensorflow as tf
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, x_test.shape)
# exit()

model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),

    # (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS).

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

model.save('saves/fashion_conv.keras')