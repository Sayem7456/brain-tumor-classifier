import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Verify weights file exists
if not os.path.exists("best_weights.h5"):
    st.error("CRITICAL ERROR: Model weights file not found!")
    st.error("Please ensure 'best_weights.h5' is in the same directory as this app.")
    st.stop()

# Define model architecture
def create_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    rescaling = tf.keras.layers.Rescaling(1./255)(inputs)
    normalization = tf.keras.layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.052441, 0.050176, 0.052627]
    )(rescaling)
    
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=normalization,
        weights=None
    )
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    
    channel_attention = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    channel_attention = tf.keras.layers.Dense(1, activation='sigmoid')(channel_attention)
    channel_attention = tf.keras.layers.Reshape((1, 1, -1))(channel_attention)
    attended_features = tf.keras.layers.multiply([base_model.output, channel_attention])
    
    x = tf.keras.layers.GlobalAveragePooling2D()(attended_features)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

# Load model with caching
@st.cache_resource
def load_model():
    model = create_model()
    model.load_weights('best_weights.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"MODEL LOADING ERROR: {str(e)}")
    st.error("Please check the model architecture and weights compatibility")
    st.stop()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title('Brain Tumor MRI Classification')
st.write("Classify MRI scans into: glioma, meningioma, notumor, or pituitary")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI', width=256)
        
        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        
        st.success(f"Prediction: **{predicted_class}**")
        st.success(f"Confidence: **{confidence:.2f}%**")
        
        # Show class probabilities
        st.bar_chart(
            data={k: float(v) for k, v in zip(class_names, predictions[0])},
            height=300
        )
        
    except Exception as e:
        st.error(f"PROCESSING ERROR: {str(e)}")
        st.error("Please ensure you uploaded a valid image file")