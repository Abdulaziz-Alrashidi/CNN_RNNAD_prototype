import tensorflow as tf

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),          # Input shape
    tf.keras.layers.RandomFlip('horizontal'),         # Horizontal flip
    tf.keras.layers.RandomRotation(0.2),              # Rotate ±20%
    tf.keras.layers.RandomTranslation(0.20, 0.20),    # Translate ±20% height/width
    tf.keras.layers.RandomZoom(0.20),                 # Zoom ±20%
    tf.keras.layers.RandomContrast(0.2),             # Contrast ±40%
    tf.keras.layers.RandomSaturation(0.2),            # Saturation ±20%
    tf.keras.layers.RandomBrightness(0.2),            # Brightness ±20%
    tf.keras.layers.RandomGaussianBlur(0.2),          # Blur 20% of images
    tf.keras.layers.RandomRotation(0.1),              # Rotation for subtle variation
    tf.keras.layers.RandomShear(0.15),                # Shear ±15%
])

