
# %%
from cProfile import label
import gradio as gr
import numpy as np
# import random as rn
# import os
import tensorflow as tf
import cv2

tf.config.experimental.set_visible_devices([], 'GPU')

#%% constantes
COLOR = np.array([163, 23, 252])/255.0
ALPHA = 0.8

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

def saliency_map(img):
    """
    return the normalized gradients overs the image, and also the prediction of the model
    """
    inp = tf.convert_to_tensor(
        img[None, :, :, None],
        dtype = tf.float32
    )
    inp_var = tf.Variable(inp)

    with tf.GradientTape() as tape:
        pred = model(inp_var, training=False)
        loss = pred[0][0]
    grads = tape.gradient(loss, inp_var)
    grads = tf.math.abs(grads) / (tf.math.reduce_max(tf.math.abs(grads))+1e-14)
    return grads, round(float(model(inp_var, training = False)))

#%%
def segment(image):
    # c = image
    print(image.shape)
    image = parse_image(image)
    print(image.shape)
    output = model.predict(image)
    # print(output)
    labels = {
        "Farsi" : 1-float(output),
        "Ruqaa" : float(output)
    }
    grads, _ = saliency_map(image[0, :, :, 0])
    s_map = grads.numpy()[0, :, :, 0]
    reconstructed_image = cv2.cvtColor(image.squeeze(0), cv2.COLOR_GRAY2RGB)
    for i in range(reconstructed_image.shape[0]):
        for j in range(reconstructed_image.shape[1]):
            reconstructed_image[i, j, :] = reconstructed_image[i, j, :] * (1-ALPHA) + s_map[i, j]* COLOR * ALPHA
    # reconstructed_image = reconstructed_image.astype(np.uint8)
    V = reconstructed_image
    # print("i shape:", i.shape)
    # print("type(i):", type(i))
    return labels, reconstructed_image

iface = gr.Interface(fn=segment, 
                    description="""
                    This is an Arab Calligraphy Style Recognition. 
                    This model predicts the style (binary classification) of the image. 
                    The model also outputs the Saliency map.
                    """,
                    inputs="image", 
                    outputs=[
                        gr.outputs.Label(num_top_classes=2, label="Style"), 
                        gr.outputs.Image(label = "Saliency map")
                    ],
                    examples=[["images/Farsi_1.jpg"], 
                              ["images/Farsi_2.jpg"],
                              ["images/real_Farsi.jpg"],
                              ["images/Ruqaa_1.jpg"],
                              ["images/Ruqaa_2.jpg"],
                              ["images/Ruqaa_3.jpg"],
                              ["images/real_Ruqaa.jpg"],
                    ]).launch()
# %%
