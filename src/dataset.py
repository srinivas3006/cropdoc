import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore

def get_datasets(data_dir, batch_size=32, img_size=(150, 150), color_mode="rgb"):
    """
    Creates tf.data.Dataset objects for training, validation, and testing.
    Automatically splits the data inside the `train` folder (80% train, 20% validation)
    so the user doesn't have to manually create a validation folder.
    """
    train_dir = os.path.join(data_dir, "train")

    # Map rgb/grayscale to the expected arguments for image_dataset_from_directory
    color_mode_arg = "grayscale" if color_mode == "grayscale" else "rgb"

    # Preprocessing layer to normalize pixel values from [0, 255] to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # 80% for training
    train_ds = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        color_mode=color_mode_arg,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
    )

    # 20% for validation
    val_ds = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
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
