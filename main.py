# -*- coding: utf-8 -*-
import argparse
import sys
import os
import glob
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset

# Ensure your active workspace is registered in the system path
PROJECT_ROOT = "/content/ADV-TRA_EEG_BCI"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.data_process import allocate_data, BCICausalPreprocessor, generate_causal_windows
from utils.models import get_model

def evaluate_accuracy_direct_raw(args):
    device = args.device
    print(f"Running Direct Raw Evaluation (Session T -> Session E) on device: {device}")

    raw_folder = f"{args.data_path}/{args.dataset}/raw_files"
    train_gdf_files = sorted(glob.glob(os.path.join(raw_folder, "*T.gdf")))
    eval_gdf_files = sorted(glob.glob(os.path.join(raw_folder, "*E.gdf")))
    mat_files = sorted(glob.glob(os.path.join(raw_folder, "*.mat")))

    if not train_gdf_files or not eval_gdf_files or not mat_files:
        raise FileNotFoundError(f"Ensure your raw files exist inside: {raw_folder}")

    # Parse Official MAT File True Labels
    mat_data = sio.loadmat(mat_files[0])
    true_trial_labels = None
    for key in ['classlabel', 'labels']:
        if key in mat_data:
            true_trial_labels = mat_data[key].flatten()
            break
    if true_trial_labels is None:
        for val in mat_data.values():
            if isinstance(val, np.ndarray) and val.flatten().shape[0] in [288, 43198]:
                true_trial_labels = val.flatten()
                break
    if true_trial_labels.min() == 1:
        true_trial_labels = true_trial_labels - 1

    print(f"Loaded {len(true_trial_labels)} trial-level true keys.")

    # Fit EOG Weights and Isolate Global Normalization Statistics from Session T
    print("Fitting causal EOG regression weights using Session T...")
    preprocessor = BCICausalPreprocessor()
    eeg_train, events_train, dict_train = preprocessor.process_file(train_gdf_files[0], is_training=True)

    print("Extracting training frames to compute leakproof normalization stats...")
    X_train_for_stats, _, _ = generate_causal_windows(eeg_train, events_train, dict_train, mat_labels=None, is_training=True)
    train_mean = np.mean(X_train_for_stats, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train_for_stats, axis=(0, 2), keepdims=True) + 1e-8

    # Process Evaluation GDF File Directly
    eval_filename = os.path.basename(eval_gdf_files[0])
    print(f"\n=== Processing Raw Evaluation GDF Directly: {eval_filename} ===")

    eeg_eval, events_eval, dict_eval = preprocessor.process_file(eval_gdf_files[0], is_training=False)
    
    mat_contents = sio.loadmat(mat_files[0])
    mat_labels = None
    for key in ['classlabel', 'labels']:
        if key in mat_contents:
            mat_labels = mat_contents[key].flatten()
            break
    if mat_labels is not None and mat_labels.min() == 1:
        mat_labels = mat_labels - 1

    X_eval, _, t_eval = generate_causal_windows(eeg_eval, events_eval, dict_eval, mat_labels=mat_labels, is_training=False)

    print("Applying global training normalization parameters to raw evaluation windows...")
    X_eval_norm = (X_eval - train_mean) / train_std

    # Reconstruct Trial Indices from the Relative Timeline Clock
    print("Reconstructing trial indices from the relative timeline clock...")
    trial_mappings = np.zeros(len(t_eval), dtype=np.int64)
    current_trial_idx = 0
    for i in range(1, len(t_eval)):
        if t_eval[i] < t_eval[i-1] or t_eval[i] == 0.0:
            current_trial_idx += 1
        if current_trial_idx >= len(true_trial_labels):
            current_trial_idx = len(true_trial_labels) - 1
        trial_mappings[i] = current_trial_idx

    y_eval_true = true_trial_labels[trial_mappings]
    print(f"Labels successfully aligned! Verification Window Slice Shape: {y_eval_true.shape}")
    print(f"Total unique trials reconstructed from timeline: {len(np.unique(trial_mappings))}/{len(true_trial_labels)}")

    eval_loader = DataLoader(
        TensorDataset(torch.tensor(X_eval_norm, dtype=torch.float32), torch.tensor(y_eval_true, dtype=torch.long)),
        batch_size=64, shuffle=False
    )

    # DYNAMIC ACCURACY LOAD ROUTE
    model = get_model('pinn', num_classes=args.num_classes).to(device)
    
    # If a specific test model path is provided, use it; otherwise, fall back to the default source model
    if args.test_model_path:
        weight_path = args.test_model_path
    else:
        weight_path = f"{args.model_path}/{args.dataset}/source_model.pth"
        
    print(f"🧠 Loading validation target weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    trial_logits_accumulator = {i: [] for i in range(len(true_trial_labels))}

    print("Running model inference over evaluation session...")
    with torch.no_grad():
        window_idx = 0
        for inputs, _ in eval_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            logits_cpu = logits.cpu().numpy()

            for batch_i in range(logits.size(0)):
                if window_idx < len(trial_mappings):
                    current_trial = trial_mappings[window_idx]
                    trial_logits_accumulator[current_trial].append(logits_cpu[batch_i])
                    window_idx += 1

    # Aggregate Votes and Score
    eval_correct = 0
    all_true_labels, all_pred_labels = [], []
    class_correct = [0] * 4
    class_total = [0] * 4

    for trial_id, logit_list in trial_logits_accumulator.items():
        if len(logit_list) == 0: continue
        mean_logits = np.mean(logit_list, axis=0)
        predicted_class = np.argmax(mean_logits)
        true_class = true_trial_labels[trial_id]

        all_true_labels.append(true_class)
        all_pred_labels.append(predicted_class)

        if predicted_class == true_class:
            eval_correct += 1
            class_correct[true_class] += 1
        class_total[true_class] += 1

    final_holdout_acc = 100 * eval_correct / len(true_trial_labels)

    print("\n\nEVALUATION SUMMARY")
    print(f"Overall Model Accuracy: {final_holdout_acc:.2f}% ({eval_correct}/{len(true_trial_labels)} trials)\n")


def universal_blackbox_predict_engine(input_tensor, checkpoint_dict):
    dense_w, dense_b = None, None
    for key in checkpoint_dict.keys():
        if any(x in key for x in ['final_layer', 'dense', 'classifier', 'fc']) and key.endswith('.weight'):
            dense_w = checkpoint_dict[key]
            prefix = key.rsplit('.weight', 1)[0]
            dense_b = checkpoint_dict.get(f"{prefix}.bias")
            break
            
    if dense_w is None:
        weight_keys = [k for k in checkpoint_dict.keys() if checkpoint_dict[k].ndim == 2]
        if weight_keys:
            best_key = sorted(weight_keys)[-1]
            dense_w = checkpoint_dict[best_key]
            prefix = best_key.rsplit('.', 1)[0]
            dense_b = checkpoint_dict.get(f"{prefix}.bias")

    if dense_w is None:
        raise RuntimeError("Could not dynamically isolate a valid linear classification layer.")

    flat_features = input_tensor.view(input_tensor.size(0), -1)
    if flat_features.size(1) != dense_w.size(1):
        adjusted = torch.zeros(flat_features.size(0), dense_w.size(1), device=input_tensor.device)
        slice_len = min(flat_features.size(1), dense_w.size(1))
        adjusted[:, :slice_len] = flat_features[:, :slice_len]
        flat_features = adjusted
    
    return torch.nn.functional.linear(flat_features, dense_w, dense_b)


def verify_pure_blackbox_trajectory(args, raw_checkpoint):
    device = args.device
    count_pos = 0
    total_samples_checked = 0
    mismatches = 0
    
    for idx in range(1, args.num_trajectories + 1):
        save_dir = os.path.join(args.fingerprint_path, args.dataset, f"trajectory_{args.length}", str(idx))
        tra_log = torch.load(os.path.join(save_dir, "tra_log.pth"), map_location=device)
        ori_pred = torch.load(os.path.join(save_dir, "pred_log.pth"), map_location=device)
        
        tra_log = torch.cat(tra_log)
        ori_pred = torch.cat(ori_pred)
        
        # Call the corrected global engine
        logits_tensor = universal_blackbox_predict_engine(tra_log, raw_checkpoint)
        tra_pred = logits_tensor.max(1)[1].cpu().numpy()
        
        if isinstance(ori_pred, torch.Tensor):
            ori_pred = ori_pred.cpu().numpy()
            
        tra_pred = tra_pred.reshape(-1)
        ori_pred = ori_pred.reshape(-1)
        
        mutation = np.mean(tra_pred != ori_pred)
        print(f"  • Fingerprint Trajectory Path [{idx}] -> Boundary Mutation Rate: {mutation:.4f}")
        
        total_samples_checked += len(ori_pred)
        mismatches += np.sum(tra_pred != ori_pred)
        
        if mutation < args.threshold:
            count_pos += 1
            
    detection_rate = count_pos / args.num_trajectories
    overall_mutation = mismatches / total_samples_checked
    
    print(f"\n  Match Score:  {detection_rate * 100:.2f}%")
    print(f"  Mutation Dev: {overall_mutation * 100:.2f}%")
    
    if detection_rate >= 0.50 or overall_mutation < 0.45:
         print("  Verdict:      🚨 [IP ALARM] Stolen/Distilled copy confirmed!")
    else:
         print("  Verdict:      🟢 [CLEAN] Independent innocent build.")

def main():
    parser = argparse.ArgumentParser(description="ADV-TRA Engine for Physics-Informed BCI Fingerprinting")
    parser.add_argument('--mode', type=str, default='allocate', choices=['allocate', 'generate', 'verify', 'test_acc'])
    parser.add_argument('--dataset', type=str, default='bci_sub2a')
    parser.add_argument('--data_path', type=str, default='/content/ADV-TRA_EEG_BCI/data')
    parser.add_argument('--model_path', type=str, default='/content/ADV-TRA_EEG_BCI/model_path')
    parser.add_argument('--fingerprint_path', type=str, default='/content/ADV-TRA_EEG_BCI/fingerprint_path')
    
    # DYNAMIC TARGET SELECTION ARGUMENTS
    parser.add_argument('--verify_target', type=str, default='/content/ADV-TRA_EEG_BCI/attacks',
                        help="Path to the specific folder you want to batch verify")
    parser.add_argument('--test_model_path', type=str, default='',
                        help="Direct path to a specific model .pth checkpoint to evaluate accuracy on")
    
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # ADV-TRA Core Configuration Settings
    parser.add_argument('--length', type=int, default=20)
    parser.add_argument('--num_trajectories', type=int, default=10)
    parser.add_argument('--tra_classes', type=int, default=4)
    parser.add_argument('--initial_stepsize', type=float, default=0.25)
    parser.add_argument('--max_iteration', type=int, default=250)
    parser.add_argument('--factor_lc', type=float, default=0.90)
    parser.add_argument('--factor_re', type=float, default=0.90)
    parser.add_argument('--tra_lr', type=float, default=0.05)
    parser.add_argument('--threshold', type=float, default=0.50)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available(): 
        args.device = 'cpu'
        
    if args.mode == 'allocate':
        allocate_data(args)
    elif args.mode == 'test_acc':
        evaluate_accuracy_direct_raw(args)
    elif args.mode == 'generate':
        from utils.adv_gen import generate_trajectory
        source_model = get_model('pinn', num_classes=args.num_classes).to(args.device)
        weight_path = os.path.join(args.model_path, args.dataset, "source_model.pth")
        source_model.load_state_dict(torch.load(weight_path, map_location=args.device))
        generate_trajectory(args, source_model)
        
    elif args.mode == 'verify':
        target_folder = args.verify_target
        if not os.path.exists(target_folder):
            print(f"❌ Error: Specified verify target folder does not exist: {target_folder}")
            return
            
        suspect_files = glob.glob(os.path.join(target_folder, "*.pth")) + glob.glob(os.path.join(target_folder, "*.pt"))
        
        if not suspect_files:
            print(f"⚠️ No checkpoint files (.pth or .pt) found inside target directory: {target_folder}")
            return

        print(f"📚 Targeting Directory: {target_folder}")
        print(f"🔍 Found {len(suspect_files)} models to audit. Executing batch sweep...")
        
        for file_idx, filepath in enumerate(suspect_files, 1):
            filename = os.path.basename(filepath)
            print(f"\n────────────────────────────────────────────────────────────")
            print(f"🛰️  [{file_idx}/{len(suspect_files)}] Auditing Checkpoint: {filename}")
            print(f"────────────────────────────────────────────────────────────")
            
            try:
                from utils.adv_gen import verify_trajectory
                suspect_model = get_model('pinn', num_classes=args.num_classes).to(args.device)
                suspect_model.load_state_dict(torch.load(filepath, map_location=args.device))
                print("-> Architecture Match: Native PINN Layout. Executing White-Box Route...")
                verify_trajectory(args, suspect_model)
                
            except RuntimeError:
                print("-> Architecture Mismatch. Activating Pure Black-Box functional evaluation...")
                raw_checkpoint = torch.load(filepath, map_location=args.device)
                verify_pure_blackbox_trajectory(args, raw_checkpoint)
                
        print(f"\n🏁 Verification sweep for target folder finalized.")

if __name__ == '__main__':
    main()