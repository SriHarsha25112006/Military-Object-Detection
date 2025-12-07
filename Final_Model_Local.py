# ==================================================================================
# FINAL GRANDMASTER PIPELINE (WINDOWS / LOCAL VERSION)
# ==================================================================================

import os
import sys
import shutil
import glob
import gc
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

print("🚀 STARTING FINAL PIPELINE (LOCAL WINDOWS MODE)...")

# ------------------------------------------------------------------
# [1/6] GPU CHECK
# ------------------------------------------------------------------
print("\n[1/6] Checking Hardware...")
if torch.cuda.is_available():
    print(f"   ✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("   ⛔ WARNING: No GPU detected. Training will be slow on CPU!")

# ------------------------------------------------------------------
# [2/6] DATASET SETUP (AUTO-DETECT)
# ------------------------------------------------------------------
print("\n[2/6] Verifying Dataset...")

# We assume the script is running inside or near the dataset folder
current_dir = os.getcwd()
true_root = None

# Search for the 'train' folder recursively in the current directory
for root, dirs, files in os.walk(current_dir):
    if 'train' in dirs and 'val' in dirs:
        true_root = root
        break

if not true_root:
    print(f"   ❌ Error: Could not find 'train' and 'val' folders inside {current_dir}")
    print("   👉 Make sure you extracted the dataset and are running this script from the correct folder.")
    sys.exit(1)
else:
    print(f"   ✅ Dataset Root Found: {true_root}")

# ------------------------------------------------------------------
# [3/6] SMART "FLOOR 500" BALANCING
# ------------------------------------------------------------------
print("\n[3/6] Applying 'Floor 500' Class Balancing...")
train_img_dir = os.path.join(true_root, 'train', 'images')
train_lbl_dir = os.path.join(true_root, 'train', 'labels')

# Scan counts
class_counts = {}
all_files = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt') and "_copy_" not in f]

for lbl_file in all_files:
    with open(os.path.join(train_lbl_dir, lbl_file), 'r') as f:
        lines = f.readlines()
    seen = set()
    for line in lines:
        try:
            c = int(line.split()[0])
            seen.add(c)
        except: continue
    for c in seen:
        class_counts[c] = class_counts.get(c, 0) + 1

# Clone low-count classes
files_to_clone = {k: [] for k in class_counts.keys()}
for lbl_file in all_files:
    with open(os.path.join(train_lbl_dir, lbl_file), 'r') as f:
        lines = f.readlines()
    seen = set()
    for line in lines:
        try:
            c = int(line.split()[0])
            seen.add(c)
        except: continue
    for c in seen: files_to_clone[c].append(lbl_file)

cloned_total = 0
for cls_id, count in class_counts.items():
    if count < 500:
        multiplier = int(500 / count)
        if multiplier > 1:
            print(f"   -> Boosting Class {cls_id} ({count} images) by {multiplier}x...")
            file_list = files_to_clone[cls_id]
            
            for lbl_file in file_list:
                base_name = os.path.splitext(lbl_file)[0]
                src_img = os.path.join(train_img_dir, base_name + ".jpg")
                if not os.path.exists(src_img): src_img = src_img.replace(".jpg", ".png")
                if not os.path.exists(src_img): continue
                
                src_lbl = os.path.join(train_lbl_dir, lbl_file)
                for i in range(multiplier - 1):
                    new_name = f"{base_name}_copy_{cls_id}_{i}"
                    if not os.path.exists(os.path.join(train_lbl_dir, new_name + ".txt")):
                        shutil.copy(src_lbl, os.path.join(train_lbl_dir, new_name + ".txt"))
                        shutil.copy(src_img, os.path.join(train_img_dir, new_name + os.path.splitext(src_img)[1]))
                        cloned_total += 1
print(f"   ✅ Balanced! Created {cloned_total} new synthetic samples.")

# ------------------------------------------------------------------
# [4/6] CONFIGURATION
# ------------------------------------------------------------------
print("\n[4/6] Generating Data Config...")
# Use absolute paths for Windows safety
abs_root = os.path.abspath(true_root)
data_yaml_path = os.path.join(current_dir, 'data.yaml')

data_yaml = {
    'path': abs_root,
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 12,
    'names': ['camouflage_soldier', 'weapon', 'military_tank', 'military_truck', 
              'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle', 
              'military_artillery', 'trench', 'military_aircraft', 'military_warship']
}
with open(data_yaml_path, 'w') as f: yaml.dump(data_yaml, f)

# ------------------------------------------------------------------
# [5/6] TRAINING (MODELS A & B)
# ------------------------------------------------------------------
# Setup Local Runs Directory
PROJECT_PATH = os.path.join(current_dir, 'Runs')
if not os.path.exists(PROJECT_PATH): os.makedirs(PROJECT_PATH)

# TRAIN MODEL A (NANO)
print("\n[5/6] Phase A: Model A (Nano)...")
run_name_n = 'military_nano_final'
last_pt_n = os.path.join(PROJECT_PATH, run_name_n, 'weights', 'last.pt')
best_pt_n = os.path.join(PROJECT_PATH, run_name_n, 'weights', 'best.pt')

if os.path.exists(last_pt_n):
    print("   ⚠️ Resuming Model A...")
    model_n = YOLO(last_pt_n)
    model_n.train(resume=True)
elif os.path.exists(best_pt_n):
    print("   ✅ Model A finished.")
else:
    print("   🆕 Starting Model A...")
    model_n = YOLO('yolov8n.pt')
    model_n.train(data=data_yaml_path, epochs=50, patience=15, imgsz=640, batch=16,
                  project=PROJECT_PATH, name=run_name_n, exist_ok=True,
                  box=7.5, hsv_h=0.015, degrees=10.0, mosaic=1.0, verbose=True)

# TRAIN MODEL B (SMALL)
print("\n[5/6] Phase B: Model B (Small)...")
run_name_s = 'military_small_final'
last_pt_s = os.path.join(PROJECT_PATH, run_name_s, 'weights', 'last.pt')
best_pt_s = os.path.join(PROJECT_PATH, run_name_s, 'weights', 'best.pt')

if os.path.exists(last_pt_s):
    print("   ⚠️ Resuming Model B...")
    model_s = YOLO(last_pt_s)
    model_s.train(resume=True)
elif os.path.exists(best_pt_s):
    print("   ✅ Model B finished.")
else:
    print("   🆕 Starting Model B...")
    model_s = YOLO('yolov8s.pt')
    model_s.train(data=data_yaml_path, epochs=50, patience=15, imgsz=640, batch=16,
                  project=PROJECT_PATH, name=run_name_s, exist_ok=True,
                  box=7.5, hsv_h=0.015, degrees=10.0, mosaic=1.0, verbose=True)

# ------------------------------------------------------------------
# [6/6] INFERENCE & SUBMISSION
# ------------------------------------------------------------------
print("\n[6/6] Running Inference...")
output_dir = os.path.join(current_dir, 'submission_final_safe')
if os.path.exists(output_dir): shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Resolve Weights
w_n = best_pt_n if os.path.exists(best_pt_n) else last_pt_n
w_s = best_pt_s if os.path.exists(best_pt_s) else last_pt_s

if not os.path.exists(w_n) or not os.path.exists(w_s):
    print("❌ CRITICAL: Weights missing. Training failed.")
else:
    model_n = YOLO(w_n)
    model_s = YOLO(w_s)

    test_dir = os.path.join(true_root, 'test', 'images')
    image_files = sorted(glob.glob(os.path.join(test_dir, '*')))
    
    if len(image_files) == 0:
        print(f"❌ Error: No test images found in {test_dir}")
        sys.exit(1)

    BATCH_SIZE = 50
    batches = [image_files[i:i+BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)]
    
    print(f"   Processing {len(image_files)} images...")
    weights = [1, 2, 3] # Nano, NanoTTA, Small
    iou_thr = 0.60
    
    for batch in tqdm(batches, desc="Inference"):
        try:
            # 1. Nano Standard
            res_n1 = model_n.predict(batch, imgsz=640, conf=0.15, augment=False, verbose=False)
            # 2. Nano TTA
            res_n2 = model_n.predict(batch, imgsz=800, conf=0.15, augment=True, verbose=False)
            # 3. Small Standard
            res_s1 = model_s.predict(batch, imgsz=640, conf=0.15, augment=False, verbose=False)

            for i, r_base in enumerate(res_n1):
                boxes_list, scores_list, labels_list = [], [], []
                
                def extract(res):
                    if len(res.boxes):
                        boxes_list.append(res.boxes.xyxyn.cpu().numpy().tolist())
                        scores_list.append(res.boxes.conf.cpu().numpy().tolist())
                        labels_list.append(res.boxes.cls.cpu().numpy().tolist())
                    else: boxes_list.append([]); scores_list.append([]); labels_list.append([])
                
                extract(res_n1[i])
                extract(res_n2[i])
                extract(res_s1[i])

                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.001)

                # Write File
                fname = os.path.basename(r_base.path)
                txt_name = os.path.splitext(fname)[0] + ".txt"
                with open(os.path.join(output_dir, txt_name), 'w') as f:
                    for b, s, l in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = b
                        xc, yc, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                        xc, yc = max(0, min(1, xc)), max(0, min(1, yc))
                        w, h = max(0, min(1, w)), max(0, min(1, h))
                        f.write(f"{int(l)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {s:.6f}\n")
            
            del res_n1, res_n2, res_s1
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"⚠️ Batch Error: {e}")

    # Zip
    print("\n[6/6] Zipping...")
    shutil.make_archive(os.path.join(current_dir, 'submission_final'), 'zip', output_dir)
    print(f"✅ DONE! Submission saved at: {os.path.join(current_dir, 'submission_final.zip')}")