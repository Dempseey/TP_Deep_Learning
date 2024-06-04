import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('saves/fashion_conv.keras')

_ , (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
test_images, test_labels = test_images[:20], test_labels[:20]
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i, img in enumerate(test_images):
    sureness = model.predict(tf.expand_dims(img, 0), verbose = 0)[0]
    pre_rank = np.argmax(sureness)
    print(np.round(sureness,3))
    # if not (class_names[pre_rank] == class_names[test_labels[i]]):
    print(i, img.shape, class_names[pre_rank] == class_names[test_labels[i]], sureness[pre_rank])
    print('predict:', class_names[pre_rank], '\nreel:', class_names[test_labels[i]])
    print()


# model.summary()