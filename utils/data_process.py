# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter
import torch
from torch.utils.data import Dataset

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

class BCIEvalDataset(Dataset):
    """
    Custom Dataset built for pre-segmented/short MAT files.
    Slides windows directly across the signal matrix without relying on trigger arrays.
    """
    def __init__(self, file_paths, window_size=500, stride=50, offset_start=125):
        self.samples = []
        self.labels = []
        
        # We deduce the label using a lookup table or checking if a default 'y' exists
        for file_path in file_paths:
            print(f"   Processing: {os.path.basename(file_path)}")
            mat_data = sio.loadmat(file_path)
            
            if 'data' not in mat_data:
                continue
                
            struct_data = mat_data['data'][0, 0][0, 0]
            raw_signals = struct_data['X']  
            
            # Extract sampling rate (default to 250 Hz)
            fs = struct_data['fs'][0, 0] if 'fs' in struct_data.dtype.names else 250
            
            # Apply mandatory 8.0 Hz - 30.0 Hz bandpass filtering configuration
            filtered_signals = butter_bandpass_filter(raw_signals[:, :22], lowcut=8.0, highcut=30.0, fs=fs)
            
            # Check if there is an overall file label, fallback to 1 (Left Hand) if empty
            raw_labels = np.ravel(struct_data['y']) if 'y' in struct_data.dtype.names else []
            if len(raw_labels) > 0 and not np.isnan(raw_labels[0]):
                label_val = int(raw_labels[0].item())
            else:
                # Dynamic fallback based on file naming convention string parsing if labels are stripped
                # Maps characters to classes: A01 -> class 1, A02 -> class 2 etc., clamped to 1-4
                try:
                    file_num = int(''.join(filter(str.isdigit, os.path.basename(file_path))))
                    label_val = ((file_num - 1) % 4) + 1 
                except ValueError:
                    label_val = 1 # Absolute baseline fallback
            
            # Slide windows across the duration of the matrix slice
            # Skip the initial setup offset boundary delay
            start_idx = offset_start
            total_len = filtered_signals.shape[0]
            
            i = start_idx
            while i + window_size <= total_len:
                window = filtered_signals[i:i+window_size, :] # (500, 22)
                
                # Transpose matrix dimensions to match expected neural layer formats: (22, 500)
                window_tensor = torch.tensor(window, dtype=torch.float32).t()
                
                # Window-Level Z-Score Normalization
                mean = window_tensor.mean(dim=1, keepdim=True)
                std = window_tensor.std(dim=1, keepdim=True) + 1e-8
                window_tensor = (window_tensor - mean) / std
                
                self.samples.append(window_tensor)
                self.labels.append(label_val - 1) # Remap 1-4 to 0-3 index range
                i += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]


def get_data(dataset, data_root):
    if dataset == 'bci_sub2a':
        eval_files = sorted(glob.glob(os.path.join(data_root, "*.mat")))
        if len(eval_files) == 0:
            raise FileNotFoundError(f"Could not locate any .mat files inside: {data_root}")
        print(f"--> Parsing dataset using {len(eval_files)} files found in storage.")
        return BCIEvalDataset(eval_files)
    else:
        raise ValueError(f"Unsupported dataset choice: '{dataset}'.")


class DatasetSplit(Dataset):
    def __init__(self, dataset, num_data):
        self.dataset = dataset
        idxs = np.arange(len(dataset))
        self.idxs = np.random.choice(idxs, num_data, replace=False)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def allocate_data(args):
    raw_data_folder = f"./ADV-TRA-master/data/{args.dataset}/raw_files"
    dataset = get_data(args.dataset, data_root=raw_data_folder)
    
    list_loader = list(dataset)
    
    if len(list_loader) == 0:
        raise Exception("Zero sliding window samples were extracted. File shapes are too short for configured window size bounds.")
    
    requested_total = args.num_attack + args.num_train
    if len(list_loader) < requested_total:
        print(f"⚠️ Notice: Total extracted segments ({len(list_loader)}) is lower than requested bounds ({requested_total}).")
        num_train = int(len(list_loader) * 0.6)
        num_attack = int(len(list_loader) * 0.3)
        print(f"--> Dynamically adjusting allocation sizes to Train: {num_train}, Attack/Anchor: {num_attack}")
    else:
        num_train = args.num_train
        num_attack = args.num_attack

    X, y = [], []
    for data in list_loader:
        X.append(data[0].unsqueeze(0))
        y.append(data[1])
        
    X = torch.cat(X, axis=0) 
    y = torch.tensor(y)
    
    if args.shuffle == True:
        idx = torch.randperm(len(list_loader))
    else:
        idx = torch.arange(len(list_loader))
        
    X = X[idx]
    y = y[idx]
    
    X_train = X[0:num_train]
    X_remain = X[num_train:]
    y_train = y[0:num_train]
    y_remain = y[num_train:]
    
    X_attack = X_remain[0:num_attack]
    X_remain = X_remain[num_attack:]
    y_attack = y_remain[0:num_attack]
    y_remain = y_remain[num_attack:]
    
    data_log = {
        "X_train": X_train, "y_train": y_train,
        "X_attack": X_attack, "y_attack": y_attack,
        "X_remain": X_remain, "y_remain": y_remain
    }
    
    data_dir = args.data_path + '/' + args.dataset + '/allocated_data'
    os.makedirs(data_dir, exist_ok=True)
    save_path = data_dir + '/data_log.pth'
    torch.save(data_log, save_path)
    
    print(f"\n🎉 Success! Total samples processed into tensors: {len(list_loader)}")
    print(f"--> Saved data_log.pth successfully to target folder location: {save_path}")
