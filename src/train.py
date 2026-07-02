import os
import json
import argparse
from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard  # type: ignore
from dataset import get_datasets
from model import build_model

def main():
    parser = argparse.ArgumentParser(description="Train the Tomato Irrigation Status CNN model")
    parser.add_argument('--color_mode', type=str, default='rgb', choices=['rgb', 'grayscale'], help='Color mode for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='dataset_splits', help='Path to dataset directory')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        return

    print(f"Loading {args.color_mode} datasets via tf.data...")
    train_ds, val_ds, class_names = get_datasets(args.data_dir, batch_size=args.batch_size, color_mode=args.color_mode)
    
    # Dynamically extract class names based on folders
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    print("Building model...")
    model = build_model(color_mode=args.color_mode, num_classes=num_classes)
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.0001),
        metrics=['acc']
    )

    # Modern optimization: Callbacks
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    model_path = os.path.join('saved_models', f'best_model_{args.color_mode}.keras')
    class_names_path = os.path.join('saved_models', 'class_names.json')

    # Save the class dictionary so predict.py can load it dynamically
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)

    callbacks = [
        ModelCheckpoint(filepath=model_path, monitor='val_acc', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=os.path.join('logs', f'run_{args.color_mode}'))
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to {model_path}")

if __name__ == "__main__":
    main()
