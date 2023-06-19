# Reference for model creation: Emotion Detection using OpenCV by Karan Sethi
# Dataset used: fer2013

import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# variable_definition
num_classes = 6
img_rows, img_cols = 128, 128
batch_size = 32

# dataset variables
training_data = "./fer2013/train"
validation_data = "./fer2013/validation"

# Set parameters from Image Data Generator, used to create augmented image batches
training_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create augmented image batches for training and validation data
training_generator = training_datagen.flow_from_directory(
    training_data,
    color_mode="grayscale",
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle="true",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data,
    color_mode="grayscale",
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle="true",
)

# Model Creation
model = Sequential()

# Layers

# 32 Filter Block
model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        input_shape=(img_rows, img_cols, 1),
    )
)
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        input_shape=(img_rows, img_cols, 1),
    )
)
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 64 Filter Block
model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 128 Filter Block
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 256 Filter Block
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Flattening/Vectorization Block
model.add(Flatten())
model.add(Dense(64, kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Vectorization Block
model.add(Dense(64, kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Block
model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))

# Postprocessing - Optimizing and Callback parameters

# ModelCheckpoint - monitor validation loss
checkpoint = ModelCheckpoint(
    "ExpressionModel.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

# Early Stopping - Break if model's learning rate slows down significantly
earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=3, verbose=1, restore_best_weights=True
)

# Reduce Learning Rate as model gains accuracy
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, verbose=1, min_delta=0.0001
)
callbacks = [earlystop, checkpoint, reduce_lr]

# Model Compilation

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
nb_train_samples = 24176
nb_validation_samples = 3006
epochs = 25

# Model Fitting
history = model.fit(
    training_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
)
