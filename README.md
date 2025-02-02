# Facial Emotion Recognition using CNN

## 📌 Overview
This project implements a **Facial Emotion Recognition** system using **Convolutional Neural Networks (CNNs)**. The model classifies human facial expressions into different emotions such as **happy, sad, angry, surprised, neutral, etc.**

## 📂 Project Structure
```
facial_emotion_recognition/
│-- data/                   # Directory for dataset (train/test images)
│-- models/                 # Saved trained models
│-- scripts/                 # Source code for training and evaluation
│   │-- train.py            # Training script
│   │-- evaluate.py         # Model evaluation
│   │-- predict.py          # Real-time emotion detection (optional)
│-- requirements.txt        # List of dependencies
│-- README.md               # Project documentation
```

##  Dataset
The FER2013 dataset is used for training and evaluation. It comprises 35,887 grayscale images (48x48 pixels) of human faces, labeled into seven emotion classes.

🔹 Training Set: 80% of the dataset
🔹 Validation Set: 20% of the dataset

Dataset Source: FER2013 on Kaggle

### 🔹 Download the Dataset
Run the following command to download the dataset:
```sh
kaggle datasets download -d msambare/fer2013
```
Then extract it:
```sh
unzip fer2013.zip -d data/
```

## ⚙️ Installation
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/4yamneBH/facial-emotion-recognition.git
   cd facial-emotion-recognition
   ```
2. **Create a Virtual Environment:**
   ```sh
   python -m venv env
   env\Scripts\activate  # On MAC OS: source env/bin/activate
   ```
3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

##  Model Architecture
The project implements a VGG16-based model, fine-tuned for emotion classification.

📌 Key Features:
 ✔ Pre-trained VGG16 backbone for feature extraction
 ✔ Additional dense layers for classification
 ✔ Dropout layers to prevent overfitting
 ✔ Adam optimizer and categorical crossentropy loss

Model Summary

```sh
model = tf.keras.models.Sequential([
    vgg16_model,  # Pre-trained VGG16 model
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emotion classes
])
```
- Uses a **CNN-based architecture** inspired by **VGG16**
- Includes **data augmentation** and **dropout layers** to prevent overfitting
- Optimized with **Adam optimizer** and **categorical cross-entropy loss**

## Training the Model
To train the CNN model, run:
```sh
python src/train.py
```

## Results & Performance
Model	Validation Accuracy	Training Accuracy
Model	Validation Accuracy	Training Accuracy
Custom CNN	~38%	~50%
VGG16 (Fine-tuned)	~70%	~85%
🔹 VGG16 outperformed the custom CNN due to its superior feature extraction capabilities.
🔹 The training loss and validation loss stabilized over epochs, reducing overfitting.

🌟 Features
✔ Real-time Emotion Detection using images
✔ User-Friendly Web Interface built with Streamlit
✔ Improved Accuracy with Transfer Learning
✔ Robust Preprocessing Pipeline to enhance model performance

##  Real-Time Emotion Detection 
To use the model for real-time emotion recognition via webcam:
```sh
python src/predict.py
```
![Logo](./exp.png)
# Facial Emotion Recognition App

This app uses a pre-trained VGG16-based model to classify facial emotions into one of seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.

## Features
- Upload an image and get emotion predictions.
- Displays confidence scores for each emotion class.
- 
Run the app using
```sh
streamlit run app.py
```
## License
This project is open-source and available under the **MIT License**.

---
💡 **Contributions & Feedback** are welcome! Feel free to open an issue or pull request. 

