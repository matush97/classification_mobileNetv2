# https://deeplizard.com/learn/video/FNqp4ZY0wDY
# https://deeplizard.com/learn/video/Zrt76AIbeh4

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

import os
import random
import shutil
import json
from function import plot_confusion_matrix, plot_history_training

# constants
not_hold = "not_hold"
one_hand_hold = "one_hand_hold"
two_hand_hold = "two_hand_hold"

# Organize data into train, valid, test dirs
os.chdir('dataset/steering-wheel-1')
if os.path.isdir('train/not_hold/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in [not_hold, one_hand_hold, two_hand_hold]:
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 200)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 100)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
os.chdir('../..')

# Process the Data
train_path = 'dataset/steering-wheel-1/train'
valid_path = 'dataset/steering-wheel-1/valid'
test_path = 'dataset/steering-wheel-1/test'

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=10)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

print(test_batches.class_indices)

# Build The fine-tuned model
mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
mobile.summary()

x = mobile.layers[-6].output
x = tf.keras.layers.Flatten()(x)
output = Dense(units=3, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-23]:
    layer.trainable = False  # not trainable

model.summary()

# Train model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_batches,
                    validation_data=valid_batches,
                    epochs=50,
                    verbose=2
                    )

# Trainig graph
print(history.history.keys())
plot_history_training(history)

# predict sign language digits
test_labels = test_batches.classes
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))  # confusion matrix

print(test_batches.class_indices)
class_indices = test_batches.class_indices

cm_plot_labels = [not_hold, one_hand_hold, two_hand_hold]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# Ulozenie modelu
model_json = model.to_json()
with open('train_result/mobileNet_steering_wheel_1_v5.json', 'w') as json_file:
    json_file.write(model_json)

from keras.models import save_model

network_saved = save_model(model, 'train_result/mobileNet_steering_wheel_1_v5.hdf5')

# Custom class.json
class_indices_dictionary = {}
i = 0

for class_item in class_indices:
    class_indices_dictionary[i] = [str(class_item)]
    i += 1

with open('train_result/class_index_steering_wheel_1_v5.json', 'w') as file:
    file.write(json.dumps(class_indices_dictionary))
