import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models\emotion_model_vgg16.h5")
labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Open webcam
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0  # Normalize
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
        
        # Convert grayscale to RGB if the model expects 3 channels
        if model.input_shape[-1] == 3:
            roi = np.repeat(roi, 3, axis=-1)
        
        prediction = model.predict(roi)
        label = labels[np.argmax(prediction)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
