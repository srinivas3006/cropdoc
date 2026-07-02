# 🚀 Complete Setup Guide for a New Laptop

This guide explains step-by-step how to take this codebase, set it up on a completely new computer, prepare the dataset, and train the AI model.

---

## 📂 Folder Structure Guide
Before you start, it's important to understand how the codebase is organized.

**Folders YOU need to create manually:**
* `dataset_original/` (You must create this and place your raw images inside it, separated into 4 class sub-folders)

**Folders AUTOMATICALLY created by the code:**
* `dataset_splits/` (Created by `dataset_splits.py`, contains the 80/15/5 math splits for train/validation/test)
* `saved_models/` (Created by `src/train.py`, stores your trained `.keras` AI brains and `class_names.json`)
* `logs/` (Created by `src/train.py`, stores TensorBoard graphs)

---
## Step 1: Install Prerequisites
Before you begin, ensure the new computer has the following installed:
1. **Python 3.11** (Make sure to check "Add Python to PATH" during installation)
2. **Git** (To clone the repository)

## Step 2: Download the Code and Install Requirements
1. Open your terminal (Command Prompt or PowerShell).
2. Clone the repository to your new laptop:
   ```bash
   git clone <YOUR_GITHUB_REPO_LINK>
   cd CropDoc
   ```
3. Install all the required Python libraries for the AI:
   ```bash
   pip install -r requirements.txt
   ```

## Step 3: Prepare the Raw Dataset Folders
The AI needs raw images to learn from. 
1. Inside the `CropDoc` folder, create a new folder named exactly: `dataset_original`
2. Inside `dataset_original`, create 4 sub-folders representing the irrigation statuses from the IEEE paper:
   * `Fully_Irrigated`
   * `Highly_Irrigated`
   * `Medium_Irrigated`
   * `Low_Irrigated`
3. **Copy your raw `.jpg` tomato leaf images into these 4 folders accordingly.**

*Your folder structure should look like this:*
```text
CropDoc/
│── dataset_original/
│   ├── Fully_Irrigated/ (Place 100% field capacity images here)
│   ├── Highly_Irrigated/ (Place 75% field capacity images here)
│   ├── Medium_Irrigated/ (Place 50% field capacity images here)
│   └── Low_Irrigated/ (Place 25% field capacity images here)
```

## Step 4: Automatically Split the Dataset
Machine Learning requires splitting the data into Training (80%), Validation (15%), and Testing (5%) sets. You don't have to do this manually!
1. In your terminal, run the dataset splitter script:
   ```bash
   python dataset_splits.py
   ```
2. **Wait 1-3 minutes.** The script will automatically do the math and copy all 20,000+ images into a new folder called `dataset_splits/`. It organizes everything perfectly for TensorFlow.

## Step 5: Train the AI Model
Now that the data is ready, it's time to build the Neural Network!
1. In your terminal, run:
   ```bash
   python src/train.py
   ```
2. The AI will start learning. You will see progress bars for each Epoch.
3. Once training finishes, the best performing AI brain will be saved in the `saved_models/` folder as `best_model_rgb.keras`.

## Step 6: Test the AI on a New Image
To test if your AI accurately detects the irrigation status of a single leaf, pick an image from your `test` folder (since the AI never saw it during training) and run the predict script:
```bash
python src/predict.py --image_path "dataset_splits/test/Fully_Irrigated/some_image.jpg" --model_path "saved_models/best_model_rgb.keras"
```
The script will output the predicted irrigation status (e.g., `Fully_Irrigated`) and its confidence percentage!
