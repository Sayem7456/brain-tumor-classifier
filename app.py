import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
# Paths for weights
EFFICIENTNET_WEIGHTS_PATH = 'efficientnetb0_notop.h5'  # path to pre-trained EfficientNetB0 no-top weights
CLASSIFIER_WEIGHTS_PATH = 'best_weights.h5'           # path to your trained classifier weights

# --- MODEL DEFINITION ---
def channel_attention_model(input_shape=(*IMAGE_SIZE, 3), num_classes=len(CLASS_NAMES)):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    base_model.load_weights('efficientnetb0_notop.h5')
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    
    channel_attention = layers.GlobalAveragePooling2D()(base_model.output)
    channel_attention = layers.Dense(1, activation='sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, -1))(channel_attention)
    attended_features = layers.multiply([base_model.output, channel_attention])

    # Global average pooling and dense layers for classification
    x = layers.GlobalAveragePooling2D()(attended_features)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, output, name='channel_attention_model')
    return model

@st.cache_resource
def load_model():
    # Build the model
    model = channel_attention_model()
    # Load classifier weights
    model.load_weights(CLASSIFIER_WEIGHTS_PATH)
    return model

# --- STREAMLIT APP ---
st.title('Brain Tumor Classification with Channel Attention')

model = load_model()

uploaded_file = st.file_uploader("Upload an MRI scan...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Load image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    st.image(img, caption='Input Image', use_column_width=True)

    # Preprocess
    img_array = np.array(img).astype('float32')
    st.write("Pixel range before normalization:", img_array.min(), img_array.max())
    input_batch = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(input_batch)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0])) * 100
    label = CLASS_NAMES[idx]

    # Show results
    st.subheader(f"Prediction: **{label}** ({conf:.1f}%)")
    st.write("Probabilities:")
    st.json({name: float(p) for name, p in zip(CLASS_NAMES, preds[0])})
    st.bar_chart(preds[0])
