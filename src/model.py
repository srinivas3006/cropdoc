import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

def build_model(color_mode="rgb", num_classes=4):
    """
    Builds the lightweight ResNet-50 inspired model from the IEEE paper:
    'Deep Learning-Based Method for Irrigation Status Detection in Tomato Using Plant Leaves'
    """
    input_channels = 3 if color_mode == "rgb" else 1
    inputs = layers.Input(shape=(224, 224, input_channels))

    # Block A
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x) # Output: (28,28,16)

    # Block B
    b1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    
    b2 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(b1)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    
    b3 = layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)
    
    b4 = layers.Add()([b2, b3])
    b5 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(b4) # Output: (14,14,32)
    
    # Block C
    c1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(b5)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    
    c2 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(c1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    
    c3 = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same')(b5)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    
    c4 = layers.Add()([c2, c3])
    c5 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(c4) # Output: (7,7,64)
    
    # Block D
    d1 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(c5)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Activation('relu')(d1)
    
    d2 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Activation('relu')(d2)
    
    d3 = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same')(c5)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Activation('relu')(d3)
    
    d4 = layers.Add()([d2, d3])
    # 7x7 input, pool size 3x3, stride 2, valid padding -> (7-3)/2 + 1 = 3 -> (3,3,128)
    d5 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(d4) 
    
    # Block E
    e1 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(d5)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.Activation('relu')(e1)
    
    e2 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.Activation('relu')(e2)
    
    e3 = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same')(d5)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.Activation('relu')(e3)
    
    e4 = layers.Add()([e2, e3])
    # 3x3 input, pool size 3x3, valid padding -> 1x1
    e5 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='valid')(e4) # Output: (1,1,256)
    
    # Output Block
    f1 = layers.Flatten()(e5)
    f2 = layers.Dense(64, activation='relu')(f1)
    f3 = layers.Dense(num_classes, activation='softmax')(f2)
    
    model = models.Model(inputs=inputs, outputs=f3)
    return model
