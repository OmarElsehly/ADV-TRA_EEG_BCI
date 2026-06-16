# BCI Intellectual Property Fingerprinting & Verification Framework

This repository implements a non-intrusive IP protection framework tailored for **Physics-Informed Neural Networks (PINNs)** classifying motor imagery EEG signals. By generating stable, adversarial trajectory fingerprints locked into the model's underlying cortical dynamics, we create a robust "DNA watermark" capable of proving model ownership even after aggressive fine-tuning attacks.

---

## 🛠️ Execution Pipeline Overview

### Step 1: Environment & Directory Setup
Initializes a clean file tree to isolate raw data matrices, training allocations, model checkpoints, and generated fingerprint keys.

* **What it does:** Programmatically creates the `ADV-TRA-master` folder ecosystem.

### Step 2: Live Dataset Procurement & Loading
Automates the retrieval of true, uncorrupted multi-class brainwave arrays.

* **What it does:** Uses `kagglehub` to download the authentic **BCI Competition IV-2a dataset** (comprising Left Hand, Right Hand, Both Feet, and Tongue motor imagery tasks) and populates the project's raw data cache.
* **Code Implementation:**

```python
import os
import shutil
import kagglehub

# 1. Download the complete dataset (Signals + Triggers)
print("--> Downloading complete BCI dataset...")
path = kagglehub.dataset_download("thngdngvn/bci-competition-iv-data-sets-2a")
print("Downloaded to temporary cache at:", path)

# 2. Target raw files directory in your ADV-TRA structure
target_dir = "./ADV-TRA-master/data/bci_sub2a/raw_files/"
os.makedirs(target_dir, exist_ok=True)

# 3. Clean up any empty label-only files from before to prevent mixing
for file_name in os.listdir(target_dir):
    file_path = os.path.join(target_dir, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

# 4. Copy the real signal files into your raw_files project folder
print(f"--> Moving files to project folder...")
for file_name in os.listdir(path):
    source_file = os.path.join(path, file_name)
    destination_file = os.path.join(target_dir, file_name)
    if os.path.isfile(source_file) and file_name.endswith('.mat'):
        shutil.copy2(source_file, destination_file)
        print(f" Saved real signal file: {file_name}")

print("\n🎉 Done! Your raw_files directory is successfully populated with the true EEG data arrays.")
```

### Step 3: Data Partitioning & Allocation

Prepares distinct training splits required for fingerprint embedding.

```bash
!python ADV-TRA-master/main.py --mode allocate --dataset bci_sub2a --num_train 1000 --num_attack 100
```

* **What it does:** Extracts and formats 1,000 distinct training windows from the raw `.mat` signal profiles into a unified PyTorch binary block (`data_log.pth`).

### Step 4: Robust Trajectory Fingerprint Generation

The core cryptographic routine of the pipeline. It creates 10 highly specific boundary-crossing sequences.

```bash
!python ADV-TRA-master/main.py --mode generate \
                               --dataset bci_sub2a \
                               --num_classes 4 \
                               --length 30 \
                               --num_trajectories 10 \
                               --initial_stepsize 0.04 \
                               --tra_lr 0.002 \
                               --max_iteration 300 \
                               --device cpu
```

* **What it does:** Runs a graph-cleared optimization loop on the CPU. It introduces micro-volt perturbations into baseline EEG slices until they smoothly cross a specific decision boundary into an incorrect class. It saves these paths as immutable watermark keys.

### Step 5: Baseline Source Verification

Establishes the ground-truth benchmark for your proprietary model.

```bash
!python ADV-TRA-master/main.py --mode verify \
                               --dataset bci_sub2a \
                               --num_classes 4 \
                               --length 30 \
                               --num_trajectories 10 \
                               --suspect_path ADV-TRA-master/model_path/bci_sub2a/source_model.pth \
                               --threshold 0.98 \
                               --device cpu
```

* **What it does:** Tests the newly minted trajectories against the un-altered `source_model.pth`. This ensures a baseline **Detection Rate of 1.00 (100%)** and a **Mutation Rate of 0.0000**, verifying that the watermark keys perfectly align with your model.

### Step 6: Adversarial Fine-Tuning Simulation

Simulates an intellectual property theft scenario where a malicious actor attempts to erase your watermark by retraining the stolen model.

* **What it does:** Loads your source weights, forces the model into gradient-active evaluation mode to handle BatchNorm batch constraints, and runs 5 aggressive Adam optimization epochs over the data pool to forcefully alter the model's weight matrices. It outputs the compromised model as `suspect_model.pth`.
* **Code Implementation:**

```python
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.models import get_model

# Path Fix
sys.path.append(os.path.abspath("ADV-TRA-master"))

print("🏋️‍♂️ Initializing suspect fine-tuning simulation...")

device = torch.device('cpu')
model = get_model('bci_sub2a', num_classes=4).to(device)
model.load_state_dict(torch.load('ADV-TRA-master/model_path/bci_sub2a/source_model.pth', map_location=device))

data_log = torch.load('ADV-TRA-master/data/bci_sub2a/allocated_data/data_log.pth', map_location=device)
X_train, y_train = data_log["X_train"].to(device), data_log["y_train"].to(device)

# Aggressive learning rate to force the weights to shift significantly
optimizer = optim.Adam(model.parameters(), lr=0.005)

# FIX: Keep model in eval mode so BatchNorm doesn't crash on batch size 1,
# but gradients will still calculate and update weights normally!
model.eval()

print("🚀 Simulating active fine-tuning over the dataset...")
for epoch in range(5):
    running_loss = 0.0
    for idx, (image, label) in enumerate(zip(X_train, y_train)):
        image = image.unsqueeze(0)
        label = label.unsqueeze(0).to(torch.long)

        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(image)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), label)

        # Backward pass changes the weights
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"   Epoch {epoch+1}/5 completed | Optimization Loss: {running_loss/len(X_train):.4f}")

output_path = "/content/suspect_model.pth"
torch.save(model.state_dict(), output_path)
print(f"\n🔒 Done! Suspect model successfully altered and saved to: {output_path}")
```

### Step 7: Final Security Audit & Verdict

Runs a definitive forensic check on the suspect network to determine if your watermark survived the fine-tuning attack.

```bash
!python ADV-TRA-master/main.py --mode verify \
                               --dataset bci_sub2a \
                               --num_classes 4 \
                               --length 30 \
                               --num_trajectories 10 \
                               --suspect_path /content/suspect_model.pth \
                               --threshold 0.98 \
                               --device cpu
```

* **What it does:** Measures how closely the suspect model's decision responses align with your saved fingerprint sequences.
* **The Verdict:** If the fingerprint detection rate remains near **1.00**, it proves that your physics-informed boundaries successfully resisted the attack, confirming unauthorized derivation and providing indisputable proof of ownership.

---

## 💾 Core Artifact Summary

When backed up, the important outputs generated by this notebook include:

* `ADV-TRA-master/fingerprint_path/bci_sub2a/trajectory_30/`: Subfolders `1` through `10` containing the unique `.pth` input trajectories and target prediction sequences.
* `/content/suspect_model.pth`: The fine-tuned test model used to validate the defense robustness.
