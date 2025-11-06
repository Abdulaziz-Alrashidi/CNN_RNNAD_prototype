import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,64,3)),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.20),
    tf.keras.layers.RandomTranslation(0.25, 0.25),
    tf.keras.layers.RandomContrast(0.30),

])