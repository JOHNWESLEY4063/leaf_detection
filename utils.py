import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data(train_dir=None, val_dir=None, img_size=(224, 224), batch_size=32, augment=False):
    """
    Load dataset with augmentation settings, error handling, and optimized batch size.
    
    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        img_size (tuple): Target image size (height, width).
        batch_size (int): Batch size for data loading.
        augment (bool): Whether to apply data augmentation to the training data.
    
    Returns:
        train_data: Training data generator (None if train_dir is not provided).
        val_data: Validation data generator.
        num_classes (int): Number of classes detected in the dataset.
    """

    # ✅ Validate dataset directories
    if train_dir and not os.path.exists(train_dir):
        print(f"⚠️ Warning: Training directory '{train_dir}' not found. Training data will be skipped.")
        train_dir = None  # Avoid crashing due to missing training data
    
    if not val_dir or not os.path.exists(val_dir):
        raise FileNotFoundError(f"❌ Error: Validation directory '{val_dir}' not found!")

    # ✅ Data Augmentation for Training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40 if augment else 0,
        width_shift_range=0.2 if augment else 0,
        height_shift_range=0.2 if augment else 0,
        shear_range=0.2 if augment else 0,
        zoom_range=0.3 if augment else 0,
        brightness_range=[0.5, 1.5] if augment else None,
        horizontal_flip=True if augment else False,
        fill_mode='nearest'
    )

    # ✅ No Augmentation for Validation Data
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = None  # Initialize training data

    # ✅ Function to calculate the safe batch size
    def safe_batch_size(directory, default_batch_size):
        """Calculate a safe batch size based on the number of images in the directory."""
        num_images = sum([len(files) for _, _, files in os.walk(directory)])
        return min(default_batch_size, max(1, num_images))  # Ensure batch size is at least 1

    # ✅ Load Training Data (if available)
    if train_dir:
        train_batch_size = safe_batch_size(train_dir, batch_size)
        try:
            train_data = train_datagen.flow_from_directory(
                train_dir,
                target_size=img_size,
                batch_size=train_batch_size,
                class_mode='categorical',
                shuffle=True
            )
            print(f"✅ Training data loaded from '{train_dir}' with batch size: {train_batch_size}")
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            train_data = None

    # ✅ Load Validation Data (Always Required)
    val_batch_size = safe_batch_size(val_dir, batch_size)
    try:
        val_data = val_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=val_batch_size,
            class_mode='categorical',
            shuffle=False
        )
        print(f"✅ Validation data loaded from '{val_dir}' with batch size: {val_batch_size}")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading validation data: {e}")

    # ✅ Retrieve Number of Classes
    num_classes = len(val_data.class_indices)
    print(f"🔍 Number of detected classes: {num_classes}")
    print(f"📌 Class Indices Mapping: {val_data.class_indices}")

    return train_data, val_data, num_classes