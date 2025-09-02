# 🔒 Face Verification System

A deep learning-based face verification system built using **TensorFlow/Keras** and **OpenCV**. This project implements a **Siamese Network** architecture to determine whether two faces belong to the same person, enabling secure and accurate facial authentication.

## 🌟 Features

- **Siamese Network Architecture**: Uses twin neural networks for robust face comparison
- **Real-time Verification**: Live camera feed integration for instant face verification
- **Custom Dataset Support**: Easy dataset creation and management
- **Model Checkpointing**: Automatic saving of training progress
- **Performance Metrics**: Comprehensive testing and evaluation tools

## 📂 Project Structure

```
face-verification-system/
│
├── data/                          # Dataset directory
│   ├── anchor/                    # Anchor images (reference/input images)
│   ├── positive/                  # Positive samples (same person as anchor)
│   └── negative/                  # Negative samples (different people)
│
├── training_checkpoints/          # Model checkpoints saved during training
│
├── application/
│   └── verification_images/       # Images for live verification testing
│
├── models/                        # Trained model files
│
├── scripts/
│   ├── file.py                    # Dataset folder creation utility
│   ├── train.py                   # Model training script
│   ├── test.py                    # Model evaluation and testing
│   └── live_test.py               # Real-time face verification
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.7+
- Webcam (for live testing)
- GPU support recommended for training

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/face-verification-system.git
cd face-verification-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Dataset Structure

```bash
# Run the setup script
python scripts/file.py

# Or create manually:
mkdir -p data/anchor data/positive data/negative
mkdir -p training_checkpoints
mkdir -p application/verification_images
```

## 📊 Dataset Preparation

### Step 1: Populate Dataset Folders

```bash
# Add images to respective folders:
# 1. data/anchor/ - Your reference images (input images)
# 2. data/positive/ - Images of the same person as anchor
# 3. data/negative/ - Images of different people (for negative samples)
```

**Dataset Guidelines:**
- **Anchor Images**: min 250 images of the target person
- **Positive Images**: min 250 images of the same person (different angles, lighting)
- **Negative Images**: min 250 images of different people

### Step 2: Verification Images Setup

```bash
# Add test images for live verification
cp your_photos/* application/verification_images/
```

## 🚀 Training the Model

### 1. Start Training

```bash
python scripts/train.py
```

**Training Parameters:**
- **Epochs**: 50-100 (adjustable)
- **Batch Size**: 16-32
- **Learning Rate**: 0.0001
- **Loss Function**: Binary Cross-entropy
- **Optimizer**: Adam

### 2. Monitor Training Progress

The training script will:
- Save model checkpoints every 10 epochs
- Display training/validation loss and accuracy
- Generate training plots and metrics

## 🧪 Testing & Evaluation

### Static Testing

```bash
python scripts/test.py
```

This will evaluate the model performance using:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **F1-Score**: Harmonic mean of precision and recall

### Live Testing

```bash
python scripts/live_test.py
```

**Live Testing Features:**
- **Real-time camera feed**
- **Press 'v'**: Capture frame for verification
- **Automatic comparison** with verification_images
- **Confidence score display** (threshold: 30/100)
- **Press 'q'**: Quit application

## 📈 Model Architecture

### Siamese Network Structure

```
Input Layer (100x100-x3) → Conv2D → MaxPool2D → Conv2D → MaxPool2D → 
Conv2D → MaxPool2D → Conv2D → Flatten → Dense (4096) → Dense (1)
```


### Folder dataset Setup

```python
# Camera controls during live testing:
# 'a' - Add frame in data/anchor
# 'p' - Add frame in data/positive
# 'q' - Quit applicatio
```

### Live Verification Setup

```python
# Camera controls during live testing:
# 'v' - Verify current frame
# 'q' - Quit application
```


## 🔧 Configuration

### Adjusting Verification Threshold

Edit `live_test.py`:

```python
detection_threshold = 0.5  # Adjust sensitivity (0.1-0.9)
verification_threshold = 0.8  # Minimum matching images required
```


## 📚 Dependencies

```txt
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request


## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) - Research paper that inspired this implementation
- Research papers on Siamese Networks for face verification
- Dataset : [https://www.kaggle.com/datasets/jessicali9530/lfw-dataset] - download

---

**⭐ Star this repository if you found it helpful!**
