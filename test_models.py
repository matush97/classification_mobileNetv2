# Importing the libraries
import numpy as np
import cv2
import tensorflow as tf
import keras.utils as image

from function import custom_decode_prediction

# constants
model_name = "mobileNet_steering_wheel_6_v2"
class_model_name = "class_index_steering_wheel_6_v2"
img_path = 'dataset/image3.jpg'

# nacitanie ulozeneho modelu
with open('train_result/' + model_name + '.json', 'r') as json_file:
    json_saved_model = json_file.read()

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('train_result/' + model_name + '.hdf5')

# Upload picture using opencv
imageShow = cv2.imread(img_path)
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expended_dims = np.expand_dims(img_array, axis=0)

# Image preprocessing
test_image = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expended_dims)
prediction = network_loaded.predict(test_image)

# Decoding of prediction
predictionLabel = custom_decode_prediction(prediction, top=1, class_list_path='train_result/' + class_model_name +
                                                                              '.json')

# Vypisanie hodnot obrazka
print("predictionLabel", predictionLabel)
print("predictionLabel[0][0]", predictionLabel[0][0])
print("predictionLabel[0][0][0]", predictionLabel[0][0][0])
print("predictionLabel[0][0][1]", predictionLabel[0][0][1])
print(
    'Nazov triedy a predikcia v percentach:  %s (%.2f%%)' % (predictionLabel[0][0][0], predictionLabel[0][0][1] * 100))

# Showing with cv2 library a picture with title and prediction in percent
imageInfo = '%s (%.2f%%)' % (predictionLabel[0][0][0], predictionLabel[0][0][1] * 100)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imageShow, imageInfo, (50, 150), font, 2.5, (0, 0, 255), 4, cv2.LINE_AA)

# Resize picture
scale_percent = 30
width = int(imageShow.shape[1] * scale_percent / 100)
height = int(imageShow.shape[0] * scale_percent / 100)
dsize = (width, height)
imageShow = cv2.resize(imageShow, dsize)

cv2.imshow('Photo', imageShow)

cv2.waitKey(0)
