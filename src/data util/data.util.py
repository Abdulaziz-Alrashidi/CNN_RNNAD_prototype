import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def get_dataset(
        data_path,
        image_size=(150,150),
        batch_size=64,
        val_split=0.30,
        label_mode="categorical",
        shuffle_buffer_size=1000,
        training_cache_path=None,
        validation_cache_path=None,
):
    """
        Load, preprocess, and split training and validation datasets from a directory.
        Requires raw data to be at a directory that have subdirectory for each class
        The function is optimized to speed up the training with caching and prefetching the data.
        If no cache path is provided, the dataset is cached in memory, fastest option but might cause issues for large datasets.
        Provide a cache path to store it in disk.


        Args:
            data_path (str): Path to the dataset directory.
            image_size (tuple[int, int]): Target size of images.
            batch_size (int): Batch size.
            val_split (float): Fraction of data to use for validation.
            label_mode (str): Label mode.
            shuffle_buffer_size (int): Buffer size for shuffling data.
            training_cache_path (str): Path to the directory for caching training data.
            validation_cache_path (str): Path to the directory for caching validation data.

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset]: Train and validation datasets.
        """
    if training_cache_path and validation_cache_path:
        if training_cache_path == validation_cache_path:
            raise ValueError("The training and validation datasets must have different path to avoid overwriting the cache.")

    train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        seed=123,
        subset='both',
        image_size=image_size,
        batch_size=batch_size,
        validation_split=val_split,
        label_mode=label_mode
    )

    if training_cache_path and validation_cache_path:
        train_dataset = train_dataset.cache(training_cache_path).shuffle(buffer_size=shuffle_buffer_size).prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.cache(validation_cache_path).prefetch(buffer_size=AUTOTUNE)
    else:
        train_dataset = train_dataset.cache().shuffle(buffer_size=shuffle_buffer_size).prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset