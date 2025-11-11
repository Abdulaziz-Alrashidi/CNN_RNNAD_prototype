def build_model(shape=(64, 64, 3), rescaling=(1.0 / 255), optimizer="adam"):
    model = tf.keras.Sequential([
        # Input, Augmentation and Scaling
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Rescaling(rescaling),

        # Block 1
        tf.keras.layers.Conv2D(32, (2, 2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(32, (2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(32, (2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(32, (2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.AveragePooling2D((2, 2)),

        # Block 2
        # tf.keras.layers.SpatialDropout2D(0.1),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # compress
        tf.keras.layers.Conv2D(32, (1, 1), padding="same"),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # expand
        tf.keras.layers.Conv2D(64, (1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # compress
        tf.keras.layers.Conv2D(32, (1, 1), padding="same"),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # expand
        tf.keras.layers.Conv2D(64, (1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.AveragePooling2D((2, 2)),


        # Block 3
        # tf.keras.layers.SpatialDropout2D(0.1),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # compress
        tf.keras.layers.Conv2D(64, (1, 1), padding="same"),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # expand
        tf.keras.layers.Conv2D(128, (1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # compress
        tf.keras.layers.Conv2D(64, (1, 1), padding="same"),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # expand
        tf.keras.layers.Conv2D(128, (1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.AveragePooling2D((2, 2)),

        # Block 4
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(600),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(400),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(200, activation="softmax"),

    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model