# Military Object Detection Project

## 1. Dataset Setup
[cite_start]**Download Link:** [Official Dataset (Google Drive)](https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format)

### **Where to keep it:**
* **For Local Use:** Create a folder named `military_object_dataset` inside your project directory and extract the files there.
    * *Structure:* `Project_Folder/military_object_dataset/train/images`
* **For Colab Use:** Upload the **zip file** (`military_object_dataset.zip`) directly to your Google Drive.
    * *Path:* `My Drive/military_object_dataset.zip`

---

## 2. How to Run

### **Scenario A: You are using Google Colab (Free GPU)**
* **File to use:** `Final_Model_Colab.py`
* **When to use:** If you do not have a powerful NVIDIA GPU on your laptop or want to run it in the cloud.
* **How to use:**
    1.  Open [Google Colab](https://colab.research.google.com/).
    2.  Upload `Final_Model_Colab.py` to the notebook or copy-paste the code into a cell.
    3.  Run the code.
    4.  It will automatically connect to your Drive, train the models, and download the `submission_final.zip`.

### **Scenario B: You are using a Local Windows/Linux PC**
* **File to use:** `Final_Model_Local.py`
* **When to use:** If you have a dedicated NVIDIA GPU (RTX 3060 or higher) and want to run everything offline.
* **How to use:**
    1.  Open your terminal/command prompt in the project folder.
    2.  Run the command:
        ```bash
        python Final_Model_Local.py
        ```
    3.  The script will auto-detect the dataset folder, train the models, and save the `submission_final.zip` in the same directory.