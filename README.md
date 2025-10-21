# 😊 Facial Emotion Recognition using CNN (Keras + TensorFlow)

This project implements a **Convolutional Neural Network (CNN)** to detect and classify human emotions from facial images.  
It uses **TensorFlow/Keras** for deep learning and can recognize key emotions such as **Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral**.

---

## 🚀 Features

- 🧠 Deep learning model built using **Keras (TensorFlow backend)**
- 🖼️ Works with grayscale facial images
- 📊 Visualizations for accuracy, loss, and confusion matrix
- ⚙️ Easily extensible for custom datasets or transfer learning
- 🧩 Preprocessing pipeline for resizing, normalization, and data augmentation

---

## 🧩 Project Structure

```
facial-emotion-recognition/
│
├── train_model.py                # Main CNN model training script
├── evaluate_model.py             # Model testing & visualization
├── model/                        # Saved model weights and architecture
├── dataset/                      # Training and validation images
├── facial_emotion_recognition_requirements.txt  # Project dependencies
└── README.md                     # Documentation
```

---

## 📦 Requirements

Make sure you have **Python 3.8+** installed.

Install dependencies:
```bash
pip install -r facial_emotion_recognition_requirements.txt
```

---

## 🧠 Dataset

You can use publicly available datasets such as:

- **[FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)** (Kaggle)
- **[CK+ Dataset](https://www.kaggle.com/datasets/shawon10/ckplus)**

Organize dataset folders as follows:

```
dataset/
│
├── train/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   ├── fear/
│   ├── disgust/
│   └── neutral/
│
└── test/
    ├── angry/
    ├── happy/
    ├── sad/
    ├── surprise/
    ├── fear/
    ├── disgust/
    └── neutral/
```

---

## ⚙️ Model Architecture (Example)

- **Input:** 48x48 grayscale image  
- **Conv2D → ReLU → MaxPooling** layers  
- **Dropout** for regularization  
- **Flatten → Dense → Softmax** output (7 emotion classes)

---

## 🧩 Example Code Snippet

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 📊 Training Example

```bash
python train_model.py
```

Monitor progress:
```
Epoch 10/50
loss: 0.6123 - accuracy: 0.7924
val_loss: 0.7321 - val_accuracy: 0.7510
```

---

## 📈 Evaluation

Visualize model performance:
```bash
python evaluate_model.py
```

- Confusion Matrix  
- Classification Report  
- Accuracy & Loss Curves  

---

## 🧠 Example Output

| Emotion | Example |
|----------|----------|
| 😀 Happy | ![happy](https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Man_Smile.jpg) |
| 😢 Sad | ![sad](https://upload.wikimedia.org/wikipedia/commons/4/4f/Man_crying.jpg) |

---

## ⚙️ Future Enhancements

- 🔁 Real-time emotion recognition using webcam (OpenCV)
- 🧩 Transfer learning with MobileNet or EfficientNet
- 🎛️ Flask or Streamlit web app integration
- 🌍 Multimodal emotion fusion (text + face)

---

## 👨‍💻 Author

**Jeevan Reddy**  
Built with ❤️ using Python, TensorFlow, and Keras.

---

## ⚖️ License

This project is released under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

---
