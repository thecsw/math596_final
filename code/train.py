# Sandy Urazayev ctu@ku.edu

# Standard Python imports
import os
import math
import numpy as np

# Exactly tensorflow/keras training imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Show images and testing
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

# M1 optimizations
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name="any")
tf.config.experimental.enable_mlir_graph_optimization()

# Some utility functions to pretty output
#from utils import *

# This is where we have our dataset (both training and validating)
images = "./small_set"

# Get some test images to evaluate the model
test_images = sorted(
    [os.path.join(images+"/test",x) for x in os.listdir(os.path.join(images, "test")) if x.endswith(".jpg")]
)

# Basic input-output relationships
image_dimension = 300
scaling_factor = 3
input_dimension = image_dimension // scaling_factor


def scale_image(image, _):
    """
    Normalize the images from [0,255]->[0,1]
    Force it to be floating value
    """
    return image / 255.0


def prepare_input(image, image_size, scaling):
    yuv = tf.image.rgb_to_yuv(image)
    y, u, v = tf.split(yuv, 3, axis=(len(yuv.shape) - 1))
    return tf.image.resize(y, [image_size, image_size], method="area")


def prepare_output(image):
    yuv = tf.image.rgb_to_yuv(image)
    y, u, v = tf.split(yuv, 3, axis=(len(yuv.shape) - 1))
    return y


# Training specifications
batch_size = 8
validation_split = 0.1
seed = 2020
checkpoint_filepath = "./checkpoints"
epochs = 5

# A very handly keras function to load images
# Load the training images, 90% go to training
train_directory = image_dataset_from_directory(
    images,
    batch_size=batch_size,
    image_size=(image_dimension, image_dimension),
    subset="training",
    validation_split=validation_split,
    seed=seed,
).map(scale_image)

# Load the validation images on checkpoints, 10% go to validation
validation_directory = image_dataset_from_directory(
    images,
    batch_size=batch_size,
    image_size=(image_dimension, image_dimension),
    subset="validation",
    validation_split=validation_split,
    seed=seed,
).map(scale_image)

# Uncomment below to check the training images are loaded
# for v in train_directory.take(1):
#     for img in v:
#         array_to_img(img).show()

# Actually scale the read images for the training purposes
train_directory = train_directory.map(
    lambda x: (prepare_input(x, input_dimension, scaling_factor), prepare_output(x))
)
# Eagerly set a buffer training for 32
train_directory = train_directory.prefetch(buffer_size=32)

# Actually scale the read images for the validation purposes
validation_directory = validation_directory.map(
    lambda x: (prepare_input(x, input_dimension, scaling_factor), prepare_output(x))
)
# Eagerly set a buffer training for 32
validation_directory = validation_directory.prefetch(buffer_size=32)

# Show the downscaled images and the bigger ones
# for batch in train_directory.take(1):
#     for img in batch[0]:
#         array_to_img(img).show()
#     for img in batch[1]:
#         array_to_img(img).show()


def build_model():
    layer_configs = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }

    input_layer = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(64, 5, **layer_configs)(input_layer)
    x = layers.Conv2D(64, 5, **layer_configs)(x)
    x = layers.Conv2D(64, 5, **layer_configs)(x)
    x = layers.Conv2D(scaling_factor ** 2, 3, **layer_configs)(x)
    output_layer = tf.nn.depth_to_space(x, scaling_factor)

    return keras.Model(input_layer, output_layer)


model = build_model()
model.summary()

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

callbacks = [early_stopping_callback, model_checkpoint_callback]
loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_function)
model.fit(
    train_directory,
    epochs=epochs,
    validation_data=validation_directory,
    callbacks=callbacks,
)
model.load_weights(checkpoint_filepath)
model.save("mymodel")
