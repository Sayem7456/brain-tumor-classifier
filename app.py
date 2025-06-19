import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define model architecture (must match training exactly)
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
    # Freeze layers except last 100
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

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title('Brain Tumor MRI Classification')
st.write("Classify MRI scans into: glioma, meningioma, notumor, or pituitary")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
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
    
    st.subheader(f"Prediction: **{predicted_class}**")
    st.subheader(f"Confidence: **{confidence:.2f}%**")
    
    # Show class probabilities
    st.bar_chart(
        data={k: v for k, v in zip(class_names, predictions[0])},
        height=300
    )