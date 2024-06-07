import tensorflow as tf
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('saves/fashion.keras')

_ , (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
test_images, test_labels = test_images[:200], test_labels[:200]
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

confusion = np.full((len(class_names), len(class_names)), 0)

for i, img in enumerate(test_images):
    sureness = model.predict(tf.expand_dims(img, 0), verbose = 0)[0]
    pre_rank = np.argmax(sureness)
    confusion[test_labels[i]][pre_rank] += 1
    
    # if not (class_names[pre_rank] == class_names[test_labels[i]]):

    # print(i, img.shape, class_names[pre_rank] == class_names[test_labels[i]], sureness[pre_rank])
    # print('predict:', class_names[pre_rank], '\nreel:', class_names[test_labels[i]])
    # print()


print(confusion)
ax = sea.heatmap(confusion, annot=True, robust=True, cmap='crest')

ax.set_xlabel('predicted')
ax.set_ylabel('truth')
plt.tight_layout()
plt.show()


# model.summary()