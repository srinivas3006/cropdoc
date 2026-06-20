import os
import json
import argparse
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

def main():
    parser = argparse.ArgumentParser(description="Predict Disease from an image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model (.keras or .h5)')
    parser.add_argument('--color_mode', type=str, default='rgb', choices=['rgb', 'grayscale'], help='Color mode used during training')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Load dynamic class names
    class_names_path = os.path.join(os.path.dirname(args.model_path), 'class_names.json')
    if not os.path.exists(class_names_path):
        print(f"Error: class_names.json not found at {class_names_path}. Ensure you trained the model first.")
        return
    with open(class_names_path, 'r') as f:
        CLASS_NAMES = json.load(f)

    print(f"Processing image {args.image_path}...")
    color_mode = "grayscale" if args.color_mode == "grayscale" else "rgb"
    img = image.load_img(args.image_path, target_size=(150, 150), color_mode=color_mode)
    
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # The normalization is handled identically to the tf.data Rescaling layer in dataset.py
    img_tensor /= 255.  
    
    print("Running prediction...")
    predictions = model.predict(img_tensor)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {CLASS_NAMES[predicted_class_index]}")
    print(f"Confidence: {confidence:.2f}%")
    
    print("\nAll Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        # ensure class index matches if the dataset loaded alphabetically
        print(f"  {class_name}: {predictions[0][i] * 100:.2f}%")

if __name__ == "__main__":
    main()
