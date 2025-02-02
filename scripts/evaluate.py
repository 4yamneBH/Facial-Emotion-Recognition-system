from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model("models/emotion_model.h5")


loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")
