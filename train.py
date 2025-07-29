import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import argparse
import datetime
from utils import load_data

# Ensure TensorFlow version is correct
if int(tf.__version__.split('.')[0]) < 2:
    raise RuntimeError(f"Error: TensorFlow version {tf.__version__} is too old. Upgrade using: pip install --upgrade tensorflow")

# Argument parsing
parser = argparse.ArgumentParser(description="Train a ResNet50 model for plant disease classification.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Initial learning rate.")
args = parser.parse_args()

# Define dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
train_dir = os.path.join(BASE_DIR, "data", "train")
val_dir = os.path.join(BASE_DIR, "data", "test")
model_path = os.path.join(BASE_DIR, "models", "best_plant_disease_model.keras")
temp_model_path = os.path.join(BASE_DIR, "models", "temp_model.keras")  # Temporary file for this session

# Load dataset
train_data, val_data, num_classes = load_data(train_dir, val_dir, batch_size=args.batch_size, augment=True)
print(f"✅ Class Indices: {train_data.class_indices}")
print(f"✅ Number of Classes: {num_classes}")

# Function to create a new model
def create_new_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-30]:  
        layer.trainable = False
    for layer in base_model.layers[-30:]:  
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Handle model loading
if os.path.exists(model_path):
    print(f"📥 Loading previous best model from {model_path}...")
    best_model = load_model(model_path)
    best_val_accuracy = best_model.evaluate(val_data)[1]  # Get validation accuracy
    print(f"📊 Current Best Validation Accuracy: {best_val_accuracy:.4f}")
else:
    print("⚠️ No previous best model found. Starting fresh.")
    best_val_accuracy = 0.0
    best_model = None

# Create or load the current model for this session
if os.path.exists(model_path):
    temp_model = load_model(model_path)
    prev_classes = temp_model.output_shape[-1]
    if prev_classes == num_classes:
        print(f"📥 Loading previous best model from {model_path}...")
        model = temp_model
    else:
        print(f"⚠️ Model class mismatch ({prev_classes} vs {num_classes}). Creating a new model.")
        os.remove(model_path)
        model = create_new_model(num_classes)
else:
    print("🔨 Creating a new model...")
    model = create_new_model(num_classes)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Callbacks
log_dir = os.path.join(BASE_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1)
model_checkpoint = ModelCheckpoint(temp_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard_callback]
    )
except tf.errors.ResourceExhaustedError:
    print("❌ Out of memory. Reducing batch size and retrying...")
    args.batch_size = max(args.batch_size // 2, 8)
    train_data, val_data, _ = load_data(train_dir, val_dir, batch_size=args.batch_size, augment=True)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard_callback]
    )

# Evaluate the best model from this session
print("🔍 Evaluating the best model from this training session...")
session_best_model = load_model(temp_model_path)
session_val_accuracy = session_best_model.evaluate(val_data)[1]

# Compare with the current best model
if session_val_accuracy > best_val_accuracy:
    print(f"🎉 New best model found! Validation Accuracy: {session_val_accuracy:.4f}")
    session_best_model.save(model_path)  # Save as the new best model
    best_val_accuracy = session_val_accuracy
else:
    print(f"❌ This session did not improve the best model. Keeping the previous best.")

# Clean up temporary file
if os.path.exists(temp_model_path):
    os.remove(temp_model_path)

# Final evaluation
print("🏆 Final Best Validation Accuracy:", best_val_accuracy)