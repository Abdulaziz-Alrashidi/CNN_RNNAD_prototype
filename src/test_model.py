import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(150, (3, 3), activation=tf.keras.activations.leaky_relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(150, (3,3), activation=tf.keras.activations.leaky_relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(150, activation=tf.keras.activations.leaky_relu),
        tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax)

    ])