import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
#import tensorflow_addons as tfa

# Verify weights file exists
if not os.path.exists("best_weights.h5"):
    st.error("Model weights file not found! Ensure 'best_weights.h5' is present")
    st.stop()

# Define model architecture (EXACTLY as in training)
def create_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    rescaling = tf.keras.layers.Rescaling(1./255)(inputs)
    normalization = tf.keras.layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.052441, 0.050176, 0.052627]
    )(rescaling)
    
    # Stem
    stem_conv = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(normalization)
    stem_bn = tf.keras.layers.BatchNormalization()(stem_conv)
    stem_activation = tf.keras.layers.Activation('relu')(stem_bn)
    
    # Full EfficientNetB0 architecture with channel attention
    # ... [ALL LAYERS FROM YOUR MODEL SUMMARY GO HERE] ...
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    base_model.load_weights('efficientnetb0_notop.h5')
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    # Final layers
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
    try:
        # Load and convert to RGB
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI', width=256)
        
        # EXACT PREPROCESSING AS IN TRAINING
        img = image.resize((224, 224))
        img_array = np.array(img).astype('float32')
        
        # Create TensorFlow dataset to match training pipeline
        # This is CRITICAL to match the same preprocessing
        ds = tf.data.Dataset.from_tensor_slices([img_array])
        ds = ds.map(lambda x: tf.image.per_image_standardization(x))
        
        # Get the preprocessed image
        for img in ds.take(1):
            preprocessed_img = img.numpy()
        
        # Expand to batch dimension
        img_array = np.expand_dims(preprocessed_img, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100
        
        st.subheader(f"Prediction: **{predicted_class}**")
        st.subheader(f"Confidence: **{confidence:.2f}%**")
        
        # Show probabilities
        st.write("Class Probabilities:")
        for i, class_name in enumerate(class_names):
            prob = predictions[0][i] * 100
            st.write(f"- {class_name}: {prob:.2f}%")
        
        # Debug output
        st.write("Raw predictions:", predictions[0])
        st.write("Predicted class index:", predicted_index)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
