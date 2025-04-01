import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("face_mask_model.h5")  # Ensure this file exists

# Class labels
labels = ["With Mask", "Without Mask"]

# Function to predict mask
def predict(img):
    img = img.convert("RGB")  # Ensure RGB format
    img = img.resize((224, 224))  # Resize for model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return labels[np.argmax(prediction)]  # Return class label

# Create Gradio interface
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text")

# Launch the Gradio interface
iface.launch()
