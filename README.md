# BCI Intellectual Property Fingerprinting & Verification Framework

This repository implements a non-intrusive IP protection framework tailored for **Physics-Informed Neural Networks (PINNs)** classifying motor imagery EEG signals. By generating stable, adversarial trajectory fingerprints locked into the model's underlying cortical dynamics, we create a robust "DNA watermark" capable of proving model ownership even after aggressive fine-tuning or distillation attacks across mismatched network architectures.

---

## 📂 Project Directory Structure

```text
/content/ADV-TRA_EEG_BCI/
├── data/
│   └── bci_sub2a/
│       ├── allocated_data/  <── PyTorch tensor data warehouse blocks
│       └── raw_files/       <── Raw GDF signal logs and MAT trial keys
├── fingerprint_path/        <── Saved boundary-crossing trace channels
├── model_path/
│   └── bci_sub2a/
│       └── source_model.pth <── Original proprietary model weights
├── attacks/                 <── Modified derivatives (fine-tuned, pruned)
└── Black_Box_Models/        <── Alternative architectures (EEGNet, CNNs)
```

---

## 🛠️ Execution Pipeline Overview

### Step 1: Environment & Directory Setup

Initializes a clean file tree to isolate raw data matrices, training allocations, model checkpoints, attacks, and generated fingerprint keys.

### Step 2: Live Dataset Procurement & Loading

Automates the retrieval of true, uncorrupted multi-class brainwave arrays.

**What it does:** Downloads the authentic BCI Competition IV-2a dataset (comprising Left Hand, Right Hand, Both Feet, and Tongue motor imagery tasks) and populates the project's raw data cache.

**Code Implementation:**

```python
import os
import shutil
import kagglehub

print("--> Downloading complete BCI dataset...")
path = kagglehub.dataset_download("thngdngvn/bci-competition-iv-data-sets-2a")
print("Downloaded to temporary cache at:", path)

target_dir = "/content/ADV-TRA_EEG_BCI/data/bci_sub2a/raw_files/"
os.makedirs(target_dir, exist_ok=True)

for file_name in os.listdir(target_dir):
    file_path = os.path.join(target_dir, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

print(f"--> Moving files to project folder...")
for file_name in os.listdir(path):
    source_file = os.path.join(path, file_name)
    destination_file = os.path.join(target_dir, file_name)
    if os.path.isfile(source_file):
        shutil.copy2(source_file, destination_file)
        print(f" Saved asset file: {file_name}")

print("\n🎉 Done! Your raw_files directory is successfully populated with the true EEG data arrays.")
```

### Step 3: Data Partitioning & Allocation

Prepares distinct training splits required for fingerprint embedding.

```bash
!python /content/ADV-TRA_EEG_BCI/main.py --mode allocate --dataset bci_sub2a
```

**What it does:** Extracts and formats distinct training windows from raw GDF signal profiles into a unified PyTorch binary block (`data_log.pth`).

### Step 4: Robust Trajectory Fingerprint Generation

The core cryptographic routine of the pipeline. It creates 10 highly specific boundary-crossing sequences.

```bash
!python /content/ADV-TRA_EEG_BCI/main.py --mode generate \
                                         --dataset bci_sub2a \
                                         --num_classes 4 \
                                         --length 20 \
                                         --num_trajectories 10 \
                                         --device cpu
```

**What it does:** Runs an optimization loop that introduces micro-volt perturbations into baseline EEG slices until they smoothly cross a specific decision boundary into an incorrect class, saving these paths as immutable watermark keys.

### Step 5: Direct Raw Evaluation

Establishes baseline performance on the target subject files directly from un-split testing partitions.

```bash
!python /content/ADV-TRA_EEG_BCI/main.py --mode test_acc --dataset bci_sub2a --device cpu
```

**What it does:** Parses evaluation records chronologically, realigns trial timings using physical timeline clocks, applies leakproof normalization stats, and scores raw multi-class target accuracy.

### Step 6: Adversarial Fine-Tuning Simulation

Simulates an intellectual property theft scenario where a malicious actor attempts to erase your watermark by retraining the stolen model on alternative subject sessions (`A03E.gdf`).

**Colab Inline Implementation:**

```python
import os
import sys
import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.data_process import BCICausalPreprocessor, generate_causal_windows
from utils.models import get_model

# Environment Configuration
PROJECT_ROOT = "/content/ADV-TRA_EEG_BCI"
sys.path.append(PROJECT_ROOT)

raw_folder = "/content/ADV-TRA_EEG_BCI/data/bci_sub2a/raw_files"
source_model_path = "/content/ADV-TRA_EEG_BCI/model_path/bci_sub2a/source_model.pth"
output_dir = "/content/ADV-TRA_EEG_BCI/attacks"

# Run Causal Processing on Target Evaluation GDF
preprocessor = BCICausalPreprocessor()
eeg_raw, events, event_dict = preprocessor.process_file(os.path.join(raw_folder, "A03E.gdf"), is_training=False)

mat_contents = sio.loadmat(os.path.join(raw_folder, "A03E.mat"))
mat_labels = None
for key in ['classlabel', 'labels']:
    if key in mat_contents:
        mat_labels = mat_contents[key].flatten()
        break
if mat_labels is not None and mat_labels.min() == 1:
    mat_labels = mat_labels - 1

X_raw, y_raw, _ = generate_causal_windows(eeg_raw, events, event_dict, mat_labels=mat_labels, is_training=False)
train_mean = np.mean(X_raw, axis=(0, 2), keepdims=True)
train_std = np.std(X_raw, axis=(0, 2), keepdims=True) + 1e-8
X_normalized = (X_raw - train_mean) / train_std

train_loader = DataLoader(TensorDataset(torch.tensor(X_normalized, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.long)), batch_size=64, shuffle=True)

# Load Source and Optimization Run
model = get_model('pinn', num_classes=4).to('cpu')
model.load_state_dict(torch.load(source_model_path, map_location='cpu'))
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        logits, loss_wc = model(inputs)
        loss = F.cross_entropy(logits, labels) + 0.1 * loss_wc
        loss.backward()
        optimizer.step()

os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, "pinn_A03E_finetuned.pth"))
print("✅ Success! A03E fine-tuned derivative saved to attacks directory.")
```

### Step 7: Targeted Batch Security Audit & Verdict

Runs a targeted forensic check on an entire directory of suspect checkpoints. It automatically matches native White-Box models or routes into an Architecture-Agnostic Black-Box pipeline if structural mismatches are detected.

**Scenario A: Auditing Stolen Derivatives (Attacks Folder)**

```bash
!python /content/ADV-TRA_EEG_BCI/main.py --mode verify \
                                         --dataset bci_sub2a \
                                         --device cpu \
                                         --verify_target /content/ADV-TRA_EEG_BCI/attacks
```

**Scenario B: Auditing Mismatched Network Topographies (Black-Box Models)**

```bash
!python /content/ADV-TRA_EEG_BCI/main.py --mode verify \
                                         --dataset bci_sub2a \
                                         --device cpu \
                                         --verify_target /content/ADV-TRA_EEG_BCI/Black_Box_Models
```

**What it does:** Iterates through every `.pth` or `.pt` file inside the targeted folder. If an architecture mismatch occurs (e.g., testing against a third-party EEGNet), the verifier bypasses rigid layers and directly maps decision responses using dynamic matrix multiplication (`torch.nn.functional.linear`).

**The Verdict:** If the fingerprint boundary mutation deviation remains low (Mutation Deviation < 45.00%), it mathematically confirms that the suspect model's decision boundaries share common descent with your proprietary network, raising an IP Alarm regardless of layer names or structural configurations.

---

## 💾 Core Artifact Summary

When backed up or synchronized, important outputs generated include:

- `/content/ADV-TRA_EEG_BCI/fingerprint_path/bci_sub2a/trajectory_20/`: Subfolders 1 through 10 containing the unique boundary trace keys.
- `/content/ADV-TRA_EEG_BCI/attacks/`: Holds fine-tuned or pruned model derivations (`pinn_A03E_finetuned.pth`).
- `/content/ADV-TRA_EEG_BCI/Black_Box_Models/`: Holds independent or distilled cross-architecture targets for blind boundary testing.
