# https://www.geeksforgeeks.org/image-recognition-with-mobilenet/
# https://www.youtube.com/watch?v=5JAZiue-fzY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=18

import tensorflow as tf
import numpy as np
import keras.utils as image
from keras.applications import imagenet_utils

# importing image
img_path = '../dataset/bird.jpg'

# initializing the model to predict the image details using predefined models.
model = tf.keras.applications.mobilenet_v2.MobileNetV2()
model.summary()

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expended_dims = np.expand_dims(img_array, axis=0)
preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expended_dims)
predictions = model.predict(preprocessed_image)

# To predict and decode the image details
results = imagenet_utils.decode_predictions(predictions)
print(results)
