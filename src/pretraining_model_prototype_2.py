import tensorflow as tf

def build_model_v2(shape=(64, 64, 3), rescaling=(1.0 / 255), optimizer="adam"):
    model = tf.keras.Sequential([
        #Input, Augmentation and Scaling
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Rescaling(rescaling),

        #Block 1
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.AveragePooling2D((2, 2)),

        #Block 2
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.AveragePooling2D((2, 2)),

        #Block 3
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.AveragePooling2D((2, 2)),

        #Block 4
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        #Block 5
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(400),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(300),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(200),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Softmax(),


    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model