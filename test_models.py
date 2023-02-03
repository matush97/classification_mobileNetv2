# Importing the libraries
import numpy as np
import cv2
import tensorflow as tf

import keras.utils as image
from keras.applications import imagenet_utils

from function import custom_decode_prediction

# nacitanie ulozeneho modelu ResNet50
with open('mobileNet_1.json', 'r') as json_file:
    json_saved_model = json_file.read()

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('mobileNet_weights_1.hdf5')

# Nacitanie obrazka
img_path = 'dataset/Sign-Language-Digits/test/1/IMG_4297.JPG'

imageShow = cv2.imread(img_path)
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expended_dims = np.expand_dims(img_array, axis=0)

# Predspracovanie obrázka
test_image = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expended_dims)
prediction = network_loaded.predict(test_image)

# Dekodovanie predikcie
# predictionLabel = imagenet_utils.decode_predictions(prediction, top=1) # problem na custom datach
# print(prediction.shape)
predictionLabel = custom_decode_prediction(prediction, top=1, class_list_path='mobileNet_1.json')


# Vypisanie hodnot obrazka
print("predictionLabel", predictionLabel)
print("predictionLabel[0][0]", predictionLabel[0][0])
print("predictionLabel[0][0][1]", predictionLabel[0][0][1])
print(
    'Nazov triedy a predikcia v percentach:  %s (%.2f%%)' % (predictionLabel[0][0][1], predictionLabel[0][0][2] * 100))

# Ukazanie pomocou kniznice cv2 obrazka s nazvom a predikciou v percentach
imageInfo = '%s (%.2f%%)' % (predictionLabel[0][0][1], predictionLabel[0][0][2] * 100)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imageShow, imageInfo, (50, 50), font, 1, (0, 255, 0), 4, cv2.LINE_AA)
cv2.imshow('Photo', imageShow)

cv2.waitKey(0)
