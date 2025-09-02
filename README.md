# 🔒 Face Verification System

A deep learning–based face verification system built using **TensorFlow/Keras** and **OpenCV**.  
This project implements a **Siamese Network** to determine whether two faces belong to the same person.  

---

## 📂 Project Structure

face-verification-system/
│
├── data/ # Dataset folder
│ ├── anchor/ # Anchor images (input images from webcam)
│ ├── positive/ # Positive images (same person as anchor)
│ └── negative/ # Negative images (different people)
│
├── training_checkpoints/ # Saved model checkpoints during training
│ └── ckpt-5 # Example checkpoint file
│
├── file.py # Script to create dataset folders
├── train.py # Script to train the face verification model
├── test.py # Script to test the trained model
├── live_test.py # Script for real-time face verification
└── README.md # Project documentation



---

## ⚙️ Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/face-verification-system.git
cd face-verification-system
