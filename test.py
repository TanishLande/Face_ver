import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall

# Disable GPU (same as training)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Data paths
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negatives")
ANC_PATH = os.path.join("data", "anchor")

# Custom L1 Distance layer (must be same as training)
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def preprocess(file_path):
    """Same preprocessing function as training"""
    byte_path = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_path)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img 

def prepare_test_data():
    """Prepare test data exactly like in training"""
    print("Loading test data...")
    
    anchor = tf.data.Dataset.list_files(ANC_PATH+"/*.jpg").take(250)
    positive = tf.data.Dataset.list_files(POS_PATH+"/*.jpg").take(250)
    negative = tf.data.Dataset.list_files(NEG_PATH+"/*.jpg").take(250)

    def assign_preprocess(input_img, validation_img, label):
        return (preprocess(input_img), preprocess(validation_img), label)

    positive_pairs = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negative_pairs = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positive_pairs.concatenate(negative_pairs)
    
    data = data.map(assign_preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    # Get test data (same split as training)
    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)
    
    return test_data

def create_model_architecture():
    """Recreate the exact model architecture from training"""
    def embedding_layer():
        inp = Input(shape=(100,100,3), name="input_image")

        # First block 
        c1 = Conv2D(64, (10,10), activation="relu")(inp)
        m1 = MaxPool2D(64, (2,2), padding="same")(c1)
        
        # Second block 
        c2 = Conv2D(128, (7,7), activation="relu")(m1)
        m2 = MaxPool2D(64, (2,2), padding="same")(c2)
        
        # Third block 
        c3 = Conv2D(128, (4,4), activation="relu")(m2)
        m3 = MaxPool2D(64, (2,2), padding="same")(c3)

        # Final block 
        c4 = Conv2D(256, (4,4), activation="relu")(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation="sigmoid")(f1)

        return Model(inputs=inp, outputs=d1, name="embedding")

    def make_siamese_model():
        input_image = Input(name="input_img", shape=(100,100,3))
        validation_image = Input(name="validation_img", shape=(100,100,3))

        embedding_model = embedding_layer()
        siamese_layer = L1Dist()
        distances = siamese_layer(embedding_model(input_image), embedding_model(validation_image))
        classifier = Dense(1, activation="sigmoid")(distances)

        return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNetwork")

    return make_siamese_model()

def load_checkpoint_5():
    """Load specific checkpoint ckpt-5"""
    print("Loading checkpoint ckpt-5...")
    
    # Create model architecture
    siamese_model = create_model_architecture()
    opt = keras.optimizers.Adam(1e-4)
    
    # Create checkpoint object
    checkpoint_dir = './training_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt-5")
    
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    
    # Load the specific checkpoint
    try:
        checkpoint.restore(checkpoint_path)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
        return siamese_model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def test_model_performance():
    """Main testing function"""
    print("="*60)
    print("SIAMESE NETWORK CHECKPOINT TESTING")
    print("="*60)
    
    # Load test data
    test_data = prepare_test_data()
    
    # Load trained model from checkpoint-5
    siamese_model = load_checkpoint_5()
    
    if siamese_model is None:
        print("Failed to load model. Exiting...")
        return
    
    print("\nModel loaded successfully! Testing performance...")
    
    # Get test batch (same as your code)
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()
    
    # Make predictions
    print("Making predictions...")
    y_hat = siamese_model.predict([test_input, test_val])
    
    # Convert to binary predictions
    res = [1 if prediction > 0.5 else 0 for prediction in y_hat]
    
    print(f"\nTest batch size: {len(y_true)}")
    print(f"Predictions shape: {y_hat.shape}")
    
    # Calculate metrics
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    
    # Recall
    m_recall = Recall()
    m_recall.update_state(y_true, res)
    recall_score = m_recall.result().numpy()
    print(f"Recall: {recall_score:.4f}")
    
    # Precision (fix: was using Recall instead of Precision)
    m_precision = Precision()
    m_precision.update_state(y_true, res)
    precision_score = m_precision.result().numpy()
    print(f"Precision: {precision_score:.4f}")
    
    # Accuracy
    accuracy = np.mean(np.array(res) == y_true)
    print(f"Accuracy: {accuracy:.4f}")
    
    # F1 Score
    if (precision_score + recall_score) > 0:
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    else:
        f1_score = 0
    print(f"F1-Score: {f1_score:.4f}")
    
    # Show prediction distribution
    print(f"\nPrediction Statistics:")
    print(f"Mean prediction score: {np.mean(y_hat):.4f}")
    print(f"Min prediction score: {np.min(y_hat):.4f}")
    print(f"Max prediction score: {np.max(y_hat):.4f}")
    
    # Show some individual predictions
    print(f"\nSample Predictions:")
    print("True | Pred | Score | Correct")
    print("-" * 30)
    for i in range(min(10, len(y_true))):
        true_val = int(y_true[i])
        pred_val = res[i]
        score = y_hat[i][0]
        correct = "✓" if pred_val == true_val else "✗"
        print(f"  {true_val}  |  {pred_val}  | {score:.3f} |   {correct}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Prediction scores distribution
    plt.subplot(1, 3, 1)
    same_scores = y_hat[y_true == 1]
    diff_scores = y_hat[y_true == 0]
    
    plt.hist(diff_scores, bins=15, alpha=0.7, label='Different pairs', color='red')
    plt.hist(same_scores, bins=15, alpha=0.7, label='Same pairs', color='blue')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.legend()
    
    # Plot 2: Confusion matrix
    plt.subplot(1, 3, 2)
    true_pos = np.sum((y_true == 1) & (np.array(res) == 1))
    false_pos = np.sum((y_true == 0) & (np.array(res) == 1))
    true_neg = np.sum((y_true == 0) & (np.array(res) == 0))
    false_neg = np.sum((y_true == 1) & (np.array(res) == 0))
    
    cm = [[true_neg, false_pos], [false_neg, true_pos]]
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=12)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks([0, 1], ['Different', 'Same'])
    plt.yticks([0, 1], ['Different', 'Same'])
    
    # Plot 3: Sample images
    plt.subplot(1, 3, 3)
    # Show accuracy by prediction confidence
    confidence_levels = np.abs(y_hat.flatten() - 0.5)
    correct_predictions = (np.array(res) == y_true)
    
    # Bin by confidence
    high_conf_mask = confidence_levels > 0.3
    med_conf_mask = (confidence_levels > 0.1) & (confidence_levels <= 0.3)
    low_conf_mask = confidence_levels <= 0.1
    
    conf_accuracies = []
    conf_labels = []
    
    if np.sum(high_conf_mask) > 0:
        conf_accuracies.append(np.mean(correct_predictions[high_conf_mask]))
        conf_labels.append('High (>0.3)')
    
    if np.sum(med_conf_mask) > 0:
        conf_accuracies.append(np.mean(correct_predictions[med_conf_mask]))
        conf_labels.append('Med (0.1-0.3)')
    
    if np.sum(low_conf_mask) > 0:
        conf_accuracies.append(np.mean(correct_predictions[low_conf_mask]))
        conf_labels.append('Low (<0.1)')
    
    plt.bar(conf_labels, conf_accuracies, color=['green', 'orange', 'red'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Confidence')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Testing completed using checkpoint ckpt-5")
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'predictions': y_hat,
        'true_labels': y_true,
        'binary_predictions': res
    }
    
    return siamese_model, results

if __name__ == "__main__":
    # Run the test
    try:
        model, results = test_model_performance()
        print("Testing completed successfully!")
        print(f"Final Results: Accuracy={results['accuracy']:.4f}, Precision={results['precision']:.4f}, Recall={results['recall']:.4f}")
    except Exception as e:
        print(f"Error during testing: {e}")