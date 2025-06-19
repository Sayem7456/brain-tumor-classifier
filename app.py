import os
import streamlit as st
import numpy as np
from PIL import Image

# Debug: Print current working directory
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir()}")

# Check TensorFlow installation
try:
    import tensorflow as tf
    st.success(f"TensorFlow version: {tf.__version__}")
except ImportError:
    st.error("TensorFlow not installed! Check requirements.txt")
    st.stop()

# Verify weights file exists
if not os.path.exists("best_weights.h5"):
    st.error("Model weights file not found! Ensure 'best_weights.h5' is present")
    st.stop()

# Define model architecture (MUST match training exactly)
def create_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    rescaling = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # IMPORTANT: Use exact normalization parameters from training
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

# Load model
@st.cache_resource
def load_model():
    try:
        model = create_model()
        model.load_weights('best_weights.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"MODEL LOADING FAILED: {str(e)}")
        st.stop()

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title('Brain Tumor MRI Classification')
st.write("Classify MRI scans into: glioma, meningioma, notumor, or pituitary")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Load and convert to RGB
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI', width=256)
        
        # Preprocessing EXACTLY as during training
        img = image.resize((224, 224))
        img_array = np.array(img).astype('float32')
        
        # Debug: Show image statistics
        st.write("Image stats - Min:", np.min(img_array), "Max:", np.max(img_array), 
                 "Mean:", np.mean(img_array, axis=(0,1)))
        
        # Expand to batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100
        
        # Debug: Show raw predictions
        st.write("Raw predictions:", predictions[0])
        
        # Show results
        st.subheader(f"Prediction: **{predicted_class}**")
        st.subheader(f"Confidence: **{confidence:.2f}%**")
        
        # Show probabilities
        st.write("### Class Probabilities:")
        for i, class_name in enumerate(class_names):
            prob = predictions[0][i] * 100
            st.write(f"- {class_name}: {prob:.2f}%")
        
        # Visualize probabilities
        st.bar_chart(
            data={k: float(v) for k, v in zip(class_names, predictions[0])},
            height=300
        )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")