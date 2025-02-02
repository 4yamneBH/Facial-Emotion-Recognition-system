import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("models/emotion_model_vgg16.h5")  # Update path if needed

# Define emotion labels (FER2013 dataset has 7 classes)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to preprocess image
def preprocess_image(image):
    try:
        # Convert to grayscale
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Resize to model input size (48x48)
        image = cv2.resize(image, (48, 48))
        # Convert grayscale to RGB by replicating the single channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Add batch dimension
        image = np.expand_dims(image, axis=0)  # Shape becomes (1, 48, 48, 3)
        # Normalize pixel values
        image = image / 255.0
        return image
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="Facial Emotion Recognition App", page_icon="😊", layout="wide")

# Title and description
st.title("😊 Facial Emotion Recognition App")
st.write("""
Upload an image of a face, and the app will predict the emotion!
""")

# Sidebar for additional information
st.sidebar.header("About")
st.sidebar.info("""
This app uses a pre-trained VGG16-based model to classify facial emotions into one of seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        with st.spinner("Processing image..."):
            processed_image = preprocess_image(image)
            if processed_image is None:
                st.error("Failed to preprocess the image. Please try another image.")
                st.stop()

        # Make a prediction
        with st.spinner("Predicting emotion..."):
            prediction = model.predict(processed_image)
            predicted_label = emotion_labels[np.argmax(prediction)]
            confidence_scores = {label: f"{score * 100:.2f}%" for label, score in zip(emotion_labels, prediction[0])}

        # Display results
        st.subheader(f"Predicted Emotion: **{predicted_label}**")
        st.write("Confidence Scores:")
        for label, score in confidence_scores.items():
            st.write(f"- {label}: {score}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to proceed.")

# Footer
st.markdown("---")
st.caption("Developed by Aymane Bouhou - [GitHub : 4ymaneBH](")