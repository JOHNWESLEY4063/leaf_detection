# evaluate.py

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("models/plant_disease_model.h5")

# Paths to validation data
val_dir = "data/validation"

# Load validation data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")