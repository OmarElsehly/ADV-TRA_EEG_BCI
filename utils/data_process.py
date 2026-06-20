# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import torch
import mne
from torch.utils.data import Dataset

class BCICausalPreprocessor:
    def __init__(self, lowcut=8.0, highcut=30.0, fs=250):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.eog_weights = None
        self.b, self.a = signal.butter(4, [self.lowcut, self.highcut], btype='bandpass', fs=self.fs)

    def fit_eog_regression(self, eeg_data, eog_data):
        eog_with_bias = np.vstack([eog_data, np.ones(eog_data.shape[1])])
        eog_cov = np.linalg.pinv(eog_with_bias @ eog_with_bias.T)
        self.eog_weights = eog_cov @ eog_with_bias @ eeg_data.T
        return self.eog_weights

    def apply_eog_regression(self, eeg_data, eog_data):
        if self.eog_weights is None:
            raise ValueError("Fit the preprocessor on training data first!")
        eog_with_bias = np.vstack([eog_data, np.ones(eog_data.shape[1])])
        return eeg_data - (self.eog_weights.T @ eog_with_bias)

    def process_file(self, filepath, is_training=True):
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose='WARNING')
        data = raw.get_data()
        np.nan_to_num(data, copy=False, nan=0.0)

        eeg_data, eog_data = data[:22, :], data[22:25, :]
        if is_training:
            self.fit_eog_regression(eeg_data, eog_data)

        eeg_clean = self.apply_eog_regression(eeg_data, eog_data)
        eeg_filtered = signal.lfilter(self.b, self.a, eeg_clean, axis=-1)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        return eeg_filtered, events, event_dict

def generate_causal_windows(eeg_filtered, events, event_dict, mat_labels=None, window_size_sec=2.0, fs=250, is_training=True):
    window_samples = int(window_size_sec * fs)
    offset_start, offset_end = int(0.5 * fs), int(3.5 * fs)
    stride = int(0.2 * fs)

    total_samples = eeg_filtered.shape[1]
    mne_id_to_gdf_str = {v: k for k, v in event_dict.items()}
    
    # Session T tracks class targets directly. Session E maps chronological '783' cues to MAT variables.
    target_events = {'769': 0, '770': 1, '771': 2, '772': 3} if is_training else {'783': -1}

    X_windows, y_labels, time_indices = [], [], []
    eval_cue_idx = 0

    for sample_idx, _, mne_event_id in events:
        gdf_event_str = mne_id_to_gdf_str.get(mne_event_id, "")
        
        if gdf_event_str in target_events:
            # Resolve true class labels sequentially for Session E tracking
            if not is_training:
                if mat_labels is not None and eval_cue_idx < len(mat_labels):
                    label = int(mat_labels[eval_cue_idx])
                else:
                    label = 0  # Fallback boundary default state
                eval_cue_idx += 1
            else:
                label = target_events[gdf_event_str]
                
            start_idx = sample_idx + offset_start
            end_idx = sample_idx + offset_end

            for t in range(start_idx + window_samples, end_idx, stride):
                if t <= total_samples:
                    window = eeg_filtered[:, t - window_samples : t]
                    if window.shape[1] == window_samples:
                        X_windows.append(window)
                        y_labels.append(label)
                        time_indices.append((t - start_idx) / fs)

    return (np.array(X_windows, dtype=np.float32),
            np.array(y_labels, dtype=np.int64),
            np.array(time_indices, dtype=np.float32))

def allocate_data(args):
    raw_data_folder = f"{args.data_path}/{args.dataset}/raw_files"
    train_gdf = sorted(glob.glob(os.path.join(raw_data_folder, "*T.gdf")))
    eval_gdf = sorted(glob.glob(os.path.join(raw_data_folder, "*E.gdf")))
    mat_files = sorted(glob.glob(os.path.join(raw_data_folder, "*.mat")))

    if not train_gdf or not eval_gdf:
        raise FileNotFoundError(f"Ensure your T and E files exist in {raw_data_folder}")

    # Step 1: Ingest trial labels safely from MAT file structures
    mat_labels = None
    if mat_files:
        print(f"Loading companion true evaluation labels: {os.path.basename(mat_files[0])}")
        mat_contents = sio.loadmat(mat_files[0])
        for key in ['classlabel', 'labels']:
            if key in mat_contents:
                mat_labels = mat_contents[key].flatten()
                break
        if mat_labels is not None and mat_labels.min() == 1:
            mat_labels = mat_labels - 1  # Standardize class layout array to 0-3 range

    print("Initializing Preprocessor Blueprint...")
    preprocessor = BCICausalPreprocessor()

    print("Processing Training Data Stream...")
    train_eeg, train_events, train_dict = preprocessor.process_file(train_gdf[0], is_training=True)
    X_train, y_train, _ = generate_causal_windows(train_eeg, train_events, train_dict, mat_labels=None, is_training=True)

    print("Processing Evaluation Data Stream with Dynamic MAT Chronological Index Mapping...")
    eval_eeg, eval_events, eval_dict = preprocessor.process_file(eval_gdf[0], is_training=False)
    X_attack, y_attack, t_attack = generate_causal_windows(eval_eeg, eval_events, eval_dict, mat_labels=mat_labels, is_training=False)

    print("Computing global training normalization statistics to prevent data leakage...")
    train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-8

    X_train_norm = (X_train - train_mean) / train_std
    X_attack_norm = (X_attack - train_mean) / train_std

    data_log = {
        "X_train": torch.tensor(X_train_norm, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_attack": torch.tensor(X_attack_norm, dtype=torch.float32),
        "y_attack": torch.tensor(y_attack, dtype=torch.long),
        "t_attack": t_attack,
        "train_mean": torch.tensor(train_mean, dtype=torch.float32),
        "train_std": torch.tensor(train_std, dtype=torch.float32),
        "eog_weights": torch.tensor(preprocessor.eog_weights, dtype=torch.float32) if preprocessor.eog_weights is not None else None
    }

    data_dir = f"{args.data_path}/{args.dataset}/allocated_data"
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data_log, f"{data_dir}/data_log.pth")
    print(f"\n🎉 Success! Extracted {X_train_norm.shape[0]} train and {X_attack_norm.shape[0]} evaluation windows.")
