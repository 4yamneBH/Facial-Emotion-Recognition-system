import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dataset Path
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='sparse'
)

# Load Pretrained VGG16 Model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(48, 48, 3))

for layer in base_model.layers:
    layer.trainable = False  

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(7, activation="softmax")(x)  # 7 emotion classes

model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),  
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,  # More epochs for better learning
    batch_size=BATCH_SIZE
)

model.save("models/emotion_model_vgg16.h5")
print("Model training complete and saved!")
