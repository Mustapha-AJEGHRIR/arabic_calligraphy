
# %%
import gradio as gr
import numpy as np
# import random as rn
# import os
import tensorflow as tf
import cv2

tf.config.experimental.set_visible_devices([], 'GPU')


#%%
def parse_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (100, 100))
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

#%%

def cnn(input_shape, output_shape):
    num_classes = output_shape[0]
    dropout_seed = 708090
    kernel_seed = 42
  
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=input_shape, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.1, seed=dropout_seed),
      tf.keras.layers.Conv2D(32, 5, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.1, seed=dropout_seed),
      tf.keras.layers.Conv2D(64, 10, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.1, seed=dropout_seed),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.Dropout(0.2, seed=dropout_seed),
      tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.Dropout(0.2, seed=dropout_seed),
      tf.keras.layers.Dense(num_classes, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed))
    ])

    return model

#%%
model = cnn((100, 100, 1), (1,))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='Adam', metrics='accuracy')

model.load_weights('weights.h5')

#%%
def segment(image):
    image = parse_image(image)
    # print(image.shape)
    output = model.predict(image)
    # print(output)
    labels = {
        "farsi" : 1-float(output),
        "ruqaa" : float(output)
    }
    return labels

iface = gr.Interface(fn=segment, 
                    inputs="image", 
                    outputs="label",
                    examples=[["images/Farsi_1.jpg"], 
                              ["images/Farsi_2.jpg"],
                              ["images/Ruqaa_1.jpg"],
                              ["images/Ruqaa_2.jpg"],
                              ["images/Ruqaa_3.jpg"],
                    ]).launch()
# %%
