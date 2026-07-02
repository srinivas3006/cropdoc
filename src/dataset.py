import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore

def get_datasets(data_dir, batch_size=32, img_size=(224, 224), color_mode="rgb"):
    """
    Creates tf.data.Dataset objects for training and validation.
    Reads from the `train` and `validation` folders in `data_dir`.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    # Map rgb/grayscale to the expected arguments for image_dataset_from_directory
    color_mode_arg = "grayscale" if color_mode == "grayscale" else "rgb"

    # Preprocessing layer to normalize pixel values from [0, 255] to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Load training dataset
    train_ds = image_dataset_from_directory(
        train_dir,
        seed=123,
        label_mode='categorical',
        color_mode=color_mode_arg,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
    )

    # Load validation dataset
    val_ds = image_dataset_from_directory(
        val_dir,
        seed=123,
        label_mode='categorical',
        color_mode=color_mode_arg,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )
    # Extract class names before the dataset is mapped and prefetched
    class_names = train_ds.class_names
    
    # Normalize datasets
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize pipeline performance with caching and prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
