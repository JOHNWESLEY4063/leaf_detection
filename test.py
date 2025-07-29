import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import load_data

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(BASE_DIR, "data", "test")
image_folder = os.path.join(BASE_DIR, "test_images")  # Folder for batch predictions
model_path = os.path.join(BASE_DIR, "models", "best_plant_disease_model.keras")  # Use .keras format

# Load trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Error: Model file not found at {model_path}")
print(f"📥 Loading best model from {model_path}...")
model = tf.keras.models.load_model(model_path, compile=False)  # Avoid auto-compilation issues
print("✅ Model loaded successfully!")

# Load validation dataset (for class labels)
train_data, val_data, num_classes = load_data(train_dir=None, val_dir=val_dir, batch_size=32, augment=False)

# Ensure class consistency
model_classes = model.output_shape[-1]
if model_classes != num_classes:
    print(f"⚠️ Warning: Model expects {model_classes} classes, but found {num_classes}. The predictions may not be accurate.")
class_indices = val_data.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Invert dictionary
print(f"✅ Class Labels: {class_labels}")

# Compile the model with appropriate metrics
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Function to predict a single image
def predict_image(image_path, confidence_threshold=50):
    """Predicts the disease category of a given image."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return

    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch

        # Make prediction
        predictions = model.predict(img_array)[0]  # Get first batch result
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Get top-3 predictions
        top_3_confidences = predictions[top_3_indices] * 100  # Convert to percentage

        # Print results
        print(f"\n🖼️ Image: {os.path.basename(image_path)}")
        for i, idx in enumerate(top_3_indices):
            class_name = class_labels.get(idx, "Unknown Class")  # Ensure no key error
            confidence = top_3_confidences[i]
            if confidence > confidence_threshold:
                print(f"  {i+1}. {class_name} ({confidence:.2f}%) ✅")
            else:
                print(f"  {i+1}. {class_name} ({confidence:.2f}%) ❌ (Low confidence)")
    except Exception as e:
        print(f"❌ Error processing image '{image_path}': {e}")

# Predict all images in a folder
def predict_batch(folder_path):
    """Predicts all images in a given folder."""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found!")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("⚠️ No valid images found in the folder!")
        return

    print(f"📂 Predicting {len(image_files)} images in '{folder_path}'...")
    for image_file in image_files:
        predict_image(os.path.join(folder_path, image_file))

# Evaluate the model on validation data
print("🔍 Evaluating model on validation set...")
try:
    loss, accuracy = model.evaluate(val_data, verbose=0)
    print(f"📊 Final Validation Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
except Exception as e:
    print(f"❌ Error during evaluation: {e}")

# Run single image prediction
test_image = os.path.join(BASE_DIR, "test.jpg")
if os.path.exists(test_image):
    predict_image(test_image)
else:
    print("⚠️ Test image 'test.jpg' not found!")

# Run batch image prediction
predict_batch(image_folder)