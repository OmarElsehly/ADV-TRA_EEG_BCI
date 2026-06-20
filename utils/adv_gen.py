# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import os
import copy
from torch.utils.data import DataLoader

def next_adv_sample(eeg_window, step_size, data_grad):
    """
    Nudges the continuous multichannel EEG window using the sign of the data gradient.
    Removed image pixel clipping ([0, 1] bounds) to preserve Z-scored voltage variations.
    """
    sign_data_grad = data_grad.sign()
    perturbed_eeg = eeg_window - step_size * sign_data_grad
    return perturbed_eeg

def capture(status, status2, stepsize_list):
    if status == 0 and status2 == 0:
        return stepsize_list
    else:
        return None

def generate_unilateral_tra(model, eeg_window, target, args, num_epoch=300, coe3=1, coe4=1, decay=0.90, lr=0.001):
    length = args.length
    half_length = int(length / 2)
    stepsize_list = [torch.tensor(args.initial_stepsize).to(args.device) - 
                     torch.tensor(0.002 * i).to(args.device) for i in range(half_length)]

    best_stepsize_list = None
    
    for epoch in range(num_epoch):
        loss_all = torch.tensor(0.0).to(args.device)
        eeg_tra = []
        grad_list = []
        eeg_tra.append(eeg_window.detach().clone())
        
        process_eeg = eeg_tra[0]
        for i in range(half_length):
            model.eval()
            model.zero_grad()
            process_eeg.requires_grad = True
            stepsize_list[i].requires_grad = False
            
            # ROUTING FIX: Enforce fingerprint_mode=True to isolate logits from physics ODE losses
            output = model(process_eeg, fingerprint_mode=True)
            
            # LOSS FIX: Swapped F.nll_loss for F.cross_entropy to handle unnormalized logit distributions
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            data_grad = process_eeg.grad.data.detach().clone()
            grad_list.append(data_grad)
            process_eeg.grad.zero_()
            
            process_eeg = next_adv_sample(process_eeg, stepsize_list[i], data_grad)
            process_eeg = process_eeg.detach().clone()
            process_eeg.requires_grad = False
            eeg_tra.append(process_eeg.detach().clone())

        pred_list = []
        status1 = 0  
        status2 = 0  
        for j in range(half_length):
            if j == 0:
                adv_eeg = eeg_tra[0]
            else:
                adv_eeg = next_adv_sample(eeg_tra[j], stepsize_list[j], grad_list[j]) 
            stepsize_list[j].requires_grad = True

            output = model(adv_eeg, fingerprint_mode=True)
            output = F.softmax(output, dim=1)
            final_pred = output.max(1, keepdim=True)[1]
            pred_list.append(final_pred[0][0].item())
            
            if j == half_length - 1:
                status1 = 0
                if final_pred != target and status2 == 0:
                    status1 = 1
                    
                best_stepsize_list = capture(status1, status2, stepsize_list)
                if epoch > num_epoch / 4 and (best_stepsize_list is not None):
                    print("Trajectory complete -> target:", target.item())
                    print(f"Status1: {status1} | Status2: {status2} | Predictions: {pred_list}")
                    return best_stepsize_list
            else: 
                if final_pred == target:
                    status2 = 1 | status2   

                loss_3 = (stepsize_list[j] * decay - stepsize_list[j + 1]) ** 2
                loss_all += loss_3 * coe3

            if stepsize_list[j] < 0:
                loss_4 = -stepsize_list[j] * coe4
                loss_all += loss_4
    
        if loss_all != 0:      
            loss_all.backward()

        for i in range(half_length):
            if stepsize_list[i].grad is not None:
                stepsize_list[i] = stepsize_list[i] - lr * stepsize_list[i].grad
                 
            if status1 == 1:
                stepsize_list[i] = stepsize_list[i] * (1 / args.factor_lc - 0.049 * (epoch / num_epoch)) 
                
            if status2 == 1:
                stepsize_list[i] = stepsize_list[i] * (args.factor_lc + 0.049 * (epoch / num_epoch)) 
                
            stepsize_list[i] = stepsize_list[i].detach().clone()
            stepsize_list[i].requires_grad = False
            
    return best_stepsize_list

def generate_bilateral_tra(model, stepsize_list, eeg_window, target):
    stepsize_list = stepsize_list + stepsize_list[::-1]
    process_eeg = eeg_window
    
    eeg_tra = []
    eeg_tra.append(process_eeg)
    grad_list = []
    
    for i in range(len(stepsize_list)):
        model.eval()
        model.zero_grad()
        process_eeg.requires_grad = True
        
        output = model(process_eeg, fingerprint_mode=True)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        data_grad = process_eeg.grad.data.detach().clone()
        grad_list.append(data_grad)
        process_eeg.grad.zero_()
        
        process_eeg = next_adv_sample(process_eeg, stepsize_list[i], data_grad)
        process_eeg = process_eeg.detach().clone()
        process_eeg.requires_grad = False
        eeg_tra.append(process_eeg.detach().clone())
        
    pred_list = []
    for adv_eeg in eeg_tra:
        output = model(adv_eeg, fingerprint_mode=True)
        output = F.softmax(output, dim=1)
        final_pred = output.max(1, keepdim=True)[1]
        pred_list.append(final_pred[0][0].item())
        
    eeg_tra = torch.cat(eeg_tra)
    return eeg_tra, pred_list

def generate_all_classes(model, eeg_window, label, args):
    class_list = np.arange(args.num_classes)
    class_list = np.delete(class_list, label.cpu())
    
    if args.tra_classes > args.num_classes:
        raise Exception("Classes traversed exceed total available network targets!")
    class_list = np.random.choice(class_list, args.tra_classes - 1, replace=False)
    
    tra_log, pred_log = [], []
    model.to(args.device)
    
    for class_i in class_list:
        target = torch.tensor(class_i).to(args.device).unsqueeze(0).to(torch.long)
        stepsize_list = generate_unilateral_tra(model, eeg_window, target, args, num_epoch=args.max_iteration, 
                                                coe3=1, coe4=1, decay=args.factor_re, lr=args.tra_lr)
        if stepsize_list is None:
            return None
            
        tra, pred = generate_bilateral_tra(model, stepsize_list, eeg_window, target)
        eeg_window = tra[-1].detach().clone().unsqueeze(0)
        
        tra_pred = model(tra, fingerprint_mode=True)
        tra_pred = tra_pred.max(1, keepdim=True)[1].reshape(-1) 
        tra_log.append(tra.detach())
        pred_log.append(copy.deepcopy(tra_pred))
    
    return tra_log, pred_log

def generate_trajectory(args, source_model):
    """
    Directly processes tensors from your allocated data warehouse (data_log.pth).
    Receives the loaded source_model object straight from main.py orchestration routing.
    """
    # LOAD ROUTING: Ingest data parameters from your high-performance BCI warehouse
    data_log_path = os.path.join(args.data_path, args.dataset, "allocated_data", "data_log.pth")
    if not os.path.exists(data_log_path):
        raise FileNotFoundError(f"Missing allocated matrix file: {data_log_path}. Run --mode allocate first.")
        
    # Tell PyTorch's security filter that NumPy array variables are safe to load
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.ndarray])
    data_log = torch.load(data_log_path, map_location=args.device, weights_only=False)
    
    # FP STRATEGY: Build fingerprints using the unseen holdout evaluation data pool (X_attack)
    X = data_log["X_attack"]
    y = data_log["y_attack"]
    
    images = X[0:2 * args.num_trajectories].to(args.device)
    labels = y[0:2 * args.num_trajectories].to(args.device)
    
    source_model.to(args.device)
    source_model.eval()
    num_finger = 0
    
    print(f"\n🚀 Initiating ADV-TRA Fingerprint Generation Sweep over {args.num_trajectories} targets...")
    for eeg_window, label in zip(images, labels):
        eeg_window = eeg_window.unsqueeze(0).to(args.device)
        label = label.unsqueeze(0).to(args.device)
        
        temp = generate_all_classes(source_model, eeg_window, label, args)
        if temp is None:
            print("⚠️ Base matrix unstable near local decision boundary, advancing to next trial...")
            continue
            
        tra_log, pred_log = temp
        num_finger += 1
        
        save_dir = os.path.join(args.fingerprint_path, args.dataset, f"trajectory_{args.length}", str(num_finger))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tra_log, os.path.join(save_dir, "tra_log.pth"))
        torch.save(pred_log, os.path.join(save_dir, "pred_log.pth"))
        print(f"📦 Cryptographic Fingerprint Subfolder [{num_finger}/{args.num_trajectories}] successfully locked.")
        
        if args.num_trajectories == num_finger:
            break

    print(f"🏁 Generation Sweep complete! Secured {num_finger} cryptographic fingerprint paths.")
    return None

def verify_trajectory(args, model):
    """
    Computes fingerprint mutation checks relative to a deployed suspect network.
    """
    model.to(args.device)
    model.eval()
    
    count_pos = 0
    print(f"\n🔍 Querying Suspect API across verified fingerprint trajectory channels...")
    for idx in range(1, args.num_trajectories + 1):
        save_dir = os.path.join(args.fingerprint_path, args.dataset, f"trajectory_{args.length}", str(idx))
        
        tra_log = torch.load(os.path.join(save_dir, "tra_log.pth"), map_location=args.device)
        ori_pred = torch.load(os.path.join(save_dir, "pred_log.pth"), map_location=args.device)
        
        tra_log = torch.cat(tra_log)
        ori_pred = torch.cat(ori_pred)
        
        # Suspect verification mapping pass
        tra_pred = model(tra_log, fingerprint_mode=True)
        tra_pred = tra_pred.max(1, keepdim=True)[1].reshape(-1)
        
        # Compute exact boundary structural deviation mutation rate
        mutation = ((tra_pred != ori_pred) * 1.0).mean().item()
        print(f"  • Fingerprint Trajectory Path [{idx}] -> Boundary Mutation Rate: {mutation:.4f}")
        
        if mutation < args.threshold:
            count_pos += 1
            
    detection_rate = count_pos / args.num_trajectories
    print(f"\n🔒 [VERIFICATION RESULT] Fingerprint verification score of suspect model: {detection_rate * 100:.2f}%")
    if detection_rate >= 0.50:
         print("🚨 [IP ALARM] Threshold breached! Suspect model confirmed as STOLEN copy.")
    else:
         print("🟢 [CLEAN] Suspect model matches baseline distribution. Innocent network.")
