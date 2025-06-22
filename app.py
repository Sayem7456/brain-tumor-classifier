import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
# Paths for weights
EFFICIENTNET_WEIGHTS_PATH = 'efficientnetb0_notop.h5'  # pretrained EfficientNetB0 no-top
CLASSIFIER_WEIGHTS_PATH = 'best_weights.h5'           # fine-tuned classifier weights

# --- MODEL DEFINITION ---
def channel_attention_model(input_shape=(*IMAGE_SIZE, 3), num_classes=len(CLASS_NAMES)):
    inputs = keras.Input(shape=input_shape)
    # Rescale inputs as done in training ([-1,1] for EfficientNet)
    x = preprocess_input(inputs)

    base_model = keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_tensor=x
    )
    base_model.load_weights(EFFICIENTNET_WEIGHTS_PATH)
    for layer in base_model.layers[:-100]:
        layer.trainable = False

    # Channel attention
    ca = layers.GlobalAveragePooling2D()(base_model.output)
    ca = layers.Dense(base_model.output_shape[-1], activation='sigmoid')(ca)
    ca = layers.Reshape((1, 1, base_model.output_shape[-1]))(ca)
    attended = layers.multiply([base_model.output, ca])

    # Classification head
    x = layers.GlobalAveragePooling2D()(attended)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='channel_attention_model')
    return model

@st.cache_resource
def load_model():
    model = channel_attention_model()
    model.load_weights(CLASSIFIER_WEIGHTS_PATH)
    return model

# --- STREAMLIT APP ---
st.title('Brain Tumor Classification with Channel Attention')

model = load_model()

uploaded_file = st.file_uploader("Upload an MRI scan...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    st.image(img, caption='Input Image', use_column_width=True)

    img_array = np.array(img).astype('float32')
    st.write("Raw range:", img_array.min(), img_array.max())

    input_batch = np.expand_dims(img_array, axis=0)
    # Apply same preprocessing as training
    input_batch = preprocess_input(input_batch)
    st.write("After preprocess_input range:", input_batch.min(), input_batch.max())

    preds = model.predict(input_batch)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0])) * 100
    label = CLASS_NAMES[idx]

    st.subheader(f"Prediction: **{label}** ({conf:.1f}%)")
    st.json({name: float(p) for name, p in zip(CLASS_NAMES, preds[0])})
    st.bar_chart(preds[0])
