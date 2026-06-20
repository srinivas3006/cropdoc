from tensorflow.keras import layers, models, regularizers  # type: ignore

def build_model(color_mode="rgb", num_classes=6):
    """
    Builds the custom Convolutional Neural Network used in the original PlantVillage project.
    
    Args:
        color_mode: 'rgb' or 'grayscale'. Defines input channels and dense layer units.
        num_classes: The number of classification categories.
    """
    input_channels = 3 if color_mode == "rgb" else 1
    
    model = models.Sequential()
    # Explicit Input layer for modern Keras 3 compatibility
    model.add(layers.Input(shape=(150, 150, input_channels)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    
    # The original project purposefully used fewer dense units for grayscale to prevent overfitting
    dense_units = 256 if color_mode == "rgb" else 128
    
    model.add(layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
