# Deep Learning-Based Method for Irrigation Status Detection in Tomato Using Plant Leaves

This project uses a Convolutional Neural Network (CNN) to automatically detect the irrigation status of tomato plants using photos of their leaves. It is built with modern Python and TensorFlow 2.x to help optimize water usage and agricultural yield.

## 🌟 Project Focus
The system classifies tomato leaf images into different irrigation states (e.g., Well-Watered vs. Drought-Stressed). By analyzing leaf morphology and visual characteristics, the deep learning model can accurately determine if the plant requires watering.

---

## 🚀 Quick Start Guide (Step-by-Step)

### Step 1: Set Up Your Data Folders
To train the model, you need a dataset of tomato leaves sorted by their irrigation status.
1. Go to `dataset_splits/train/` (and do the same for `dataset_splits/validation/`).
2. Create a folder for each irrigation status category.
3. Name the folders clearly (e.g., `Well_Watered`, `Water_Stressed`, `Drought`).
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
1. **Dynamic Counting:** The script looks at your `dataset_splits/train/` folder and counts exactly how many classes you made.
2. **Brain Building:** It configures the Neural Network to classify those specific irrigation statuses.
3. **Saving the Dictionary:** It saves a `class_names.json` file inside `saved_models/` so the system permanently remembers the exact names of your folders.
4. **Training:** It feeds all the images into the network and saves the best model to the `saved_models/` folder.

*(To view live training progress graphs, run `tensorboard --logdir logs/` in a separate terminal and open `http://localhost:6006` in your browser).*

### Step 4: Test an Unseen Image (Prediction)
Once training says `Training complete.`, you can test a single leaf image:

```bash
python src/predict.py --image_path "path/to/test_leaf.jpg" --model_path "saved_models/best_model_rgb.keras"
```

**What happens:** 
1. The script loads your trained `.keras` model.
2. It automatically reads the `class_names.json` dictionary that was generated during Step 3.
3. It analyzes your tomato leaf image and prints out the predicted irrigation status with a confidence percentage!
