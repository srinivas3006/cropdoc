# Universal CropDoc Image Classifier

This project uses a Convolutional Neural Network (CNN) to automatically detect diseases in ANY plant, vegetable, or object using photos. It is built with modern Python and TensorFlow 2.x and is **100% dynamically scalable**.

## 🌟 The Universal Capabilities
This codebase does not have any hardcoded limits. The Artificial Intelligence dynamically builds itself based on the folders you give it. 
* If you give it 2 folders of Apple images, it becomes an Apple classifier.
* If you give it 50 folders of mixed vegetables, it becomes a multi-vegetable super-classifier.

---

## 🚀 Quick Start Guide (Step-by-Step)

### Step 1: Set Up Your Data Folders
To make the model smart, you just have to give it pictures!
1. Go to `dataset_splits/train/` (and do the same for `dataset_splits/validation/`).
2. Create a brand new folder for every disease or category you want it to learn.
3. Name the folders clearly (e.g., `Potato_Healthy`, `Potato_Blight`, `Tomato_Rust`).
4. Drag and drop your `.jpg` images into the matching folders.

### Step 2: Install Requirements
Make sure you have Python 3.11 installed, then run:
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
Once your folders are set up, open your terminal and run:
```bash
python src/train.py --epochs 100 --color_mode rgb
```

**What happens:** 
1. **Dynamic Counting:** The script looks at your `dataset_splits/train/` folder and counts exactly how many folders you made.
2. **Brain Building:** It scales the Neural Network up or down to perfectly match your number of folders.
3. **Saving the Dictionary:** It saves a `class_names.json` file inside `saved_models/` so the system permanently remembers the exact names of your folders.
4. **Training:** It feeds all the images into the network and saves the best model to the `saved_models/` folder.

*(To view live training progress graphs, run `tensorboard --logdir logs/` in a separate terminal and open `http://localhost:6006` in your browser).*

### Step 4: Test an Unseen Image (Prediction)
Once training says `Training complete.`, you can test a single image:

```bash
python src/predict.py --image_path "path/to/test.jpg" --model_path "saved_models/best_model_rgb.keras"
```

**What happens:** 
1. The script loads your trained `.keras` model.
2. It automatically reads the `class_names.json` dictionary that was generated during Step 3.
3. It analyzes your image and prints out the exact folder name it belongs to with a confidence percentage!
