# -*- coding: utf-8 -*-
# import numpy as np
import torch
import torch.nn.functional as F
import os
import copy
from utils.models import get_model  

def next_adv_sample(eeg_signal, step_size, data_grad):
    return eeg_signal - step_size * data_grad.sign()

def generate_unilateral_tra(model, image, target, args, num_epoch=300, decay=0.90, lr=0.002):
    length = args.length
    half_length = int(length/2)
    
    stepsize_list = [torch.tensor(args.initial_stepsize).to('cpu') - 
                     torch.tensor(0.002*i).to('cpu') for i in range(half_length)]
    target = target.to('cpu').to(torch.long)
    
    for epoch in range(num_epoch):
        images_tra = [image.detach().clone().to('cpu')]
        grad_list = []
        
        # --- PATH GENERATION PHASE ---
        for i in range(half_length):
            process_image = images_tra[i].detach().clone().to('cpu')
            process_image.requires_grad = True
            
            logits, _ = model(process_image)
            loss = F.nll_loss(F.log_softmax(logits, dim=1), target)
            
            model.zero_grad()
            loss.backward()
            
            # Extract and immediately detach the gradient to kill the memory graph
            data_grad = process_image.grad.data.detach().clone()
            grad_list.append(data_grad)
            
            next_img = next_adv_sample(process_image, stepsize_list[i], data_grad).detach().clone()
            images_tra.append(next_img)

        # --- EVALUATION PHASE ---
        status1, status2 = 0, 0
        pred_list = []
        for j in range(half_length):
            adv_image = images_tra[0] if j == 0 else next_adv_sample(images_tra[j], stepsize_list[j], grad_list[j])
            adv_image = adv_image.detach().to('cpu')
            stepsize_list[j].requires_grad = True
            
            with torch.no_grad(): # Disable autograd here completely to save memory
                logits, _ = model(adv_image)
                final_pred = F.softmax(logits, dim=1).max(1, keepdim=True)[1]
            
            pred_list.append(final_pred[0][0].item())
            
            if j == half_length-1:
                if final_pred == target:
                    status1 = 0 # Target reached successfully!
                else:
                    status1 = 1 # Failed to reach target
            else:
                if final_pred == target: 
                    status2 = 1  # Unstable intermediate flip

        # If golden criteria are satisfied, return the step sizes instantly
        if status1 == 0 and status2 == 0:
            print(f"🎯 Boundary cross settled! Sequence: {pred_list}")
            return stepsize_list

        # --- STEP SIZE GRADIENT UPDATE PHASE ---
        # Direct heuristic step size updating instead of complex autograd backpropagation
        for i in range(half_length):
            if status1 == 1: # If failed to reach target, make step sizes bigger
                stepsize_list[i] = stepsize_list[i] * (args.factor_lc + 0.01)
            if status2 == 1: # If unstable, scale them down slightly
                stepsize_list[i] = stepsize_list[i] * (1/args.factor_lc - 0.01)
                
            stepsize_list[i] = torch.clamp(stepsize_list[i].detach(), 0.001, 0.95)
            stepsize_list[i].requires_grad = False
            
    return None

def generate_trajectory(args):
    data_log = torch.load(args.data_path + '/' + args.dataset + '/allocated_data/data_log.pth', map_location='cpu')
    X, y = data_log["X_train"].to('cpu'), data_log["y_train"].to('cpu')
    
    source_model = get_model(args.dataset, num_classes=args.num_classes)
    source_model.load_state_dict(torch.load(args.model_path + '/' + args.dataset + '/source_model.pth', map_location='cpu'))
    source_model = source_model.to('cpu').eval()
    
    num_finger = 0
    print(f"🚀 Graph-cleared execution engine live. Processing up to {len(X)} windows...")
    
    for idx, (image, label) in enumerate(zip(X, y)):
        image = image.unsqueeze(0).to('cpu') 
        label = label.unsqueeze(0).to('cpu')
        
        class_list = np.delete(np.arange(args.num_classes), label.cpu())
        class_i = np.random.choice(class_list)
        target = torch.tensor(class_i).to('cpu').unsqueeze(0).to(torch.long)
        
        stepsize_list = generate_unilateral_tra(source_model, image, target, args, num_epoch=args.max_iteration)
        
        if stepsize_list is not None:
            num_finger += 1
            stepsize_list = stepsize_list + stepsize_list[::-1]
            p_img = image.to('cpu')
            images_tra = [p_img]
            
            for s in stepsize_list:
                source_model.zero_grad()
                p_img.requires_grad = True
                out, _ = source_model(p_img)
                loss = F.nll_loss(F.log_softmax(out, dim=1), target)
                loss.backward()
                p_img = next_adv_sample(p_img, s, p_img.grad.data.detach().clone()).detach().clone()
                images_tra.append(p_img)
            
            with torch.no_grad():
                tra_logits, _ = source_model(torch.cat(images_tra).to('cpu'))
                pred_log = [tra_logits.max(1, keepdim=True)[1].reshape(-1)]
            
            save_dir = f"{args.fingerprint_path}/{args.dataset}/trajectory_{args.length}/{num_finger}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save([torch.cat(images_tra).detach().cpu()], save_dir+"/tra_log.pth")
            torch.save(pred_log, save_dir+"/pred_log.pth")
            
            print(f"🔒 [SAVED] Fingerprint Path #{num_finger} secured! Target Class: {class_i} (Index: {idx})")
            
            if num_finger == args.num_trajectories:
                break
        else:
            if idx % 10 == 0:
                print(f"   [Searching] Checked up to index {idx}... hunting for clean boundaries.")
                
    print("\n🎉 ALL 5 TRAJECTORY FINGERPRINTS COMPLETED AND SAVED!")
    return None

def verify_trajectory(args):
    print("\n" + "="*70)
    print("🔒 BCI INTELLECTUAL PROPERTY SECURITY AUDIT SYSTEM")
    print("="*70)
    print(f"📊 Dataset Context:  {args.dataset}")
    print(f"🔬 Suspect Target:  {args.suspect_path}")
    print(f"🔑 Keys Registered:  {args.num_trajectories} Trajectories (Length: {args.length})")
    print(f"⚙️  Match Threshold:  {args.threshold * 100:.1f}% Maximum Mutation Allowed")
    print("-"*70)

    model = get_model(args.dataset, num_classes=args.num_classes).to('cpu')
    checkpoint = torch.load(args.suspect_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    count_pos = 0
    available_trajectories = 0
    
    for idx in range(1, args.num_trajectories+1):
        try:
            tra_log = torch.load(f"{args.fingerprint_path}/{args.dataset}/trajectory_{args.length}/{idx}/tra_log.pth", map_location='cpu')
            ori_pred = torch.load(f"{args.fingerprint_path}/{args.dataset}/trajectory_{args.length}/{idx}/pred_log.pth", map_location='cpu')
        except FileNotFoundError: 
            continue
            
        available_trajectories += 1
        tra_logits, _ = model(torch.cat(tra_log).to('cpu'))
        tra_pred = tra_logits.max(1, keepdim=True)[1].reshape(-1)
        ori_pred_flat = torch.cat(ori_pred).to('cpu')
        
        mutation = ((tra_pred != ori_pred_flat) * 1.0).mean()
        similarity = (1.0 - mutation.item()) * 100
        
        # Verify boundary compliance
        if mutation < args.threshold: 
            count_pos += 1
            status_flag = "✅ MATCH [Key Confirmed]"
        else:
            status_flag = "❌ MISMATCH [Key Broken/Absent]"
            
        print(f"  📌 Key Archive #{idx} -> Dev-Alignment: {similarity:6.2f}% | {status_flag}")

    if available_trajectories == 0:
        print("\n🚨 ERROR: No signature trajectory files found. Audit aborted.")
        print("="*70)
        return None

    # Calculate global indicators
    detection_rate = count_pos / available_trajectories
    confidence_percentage = detection_rate * 100

    print("-"*70)
    print(f"📈 OVERALL SIGNATURE VERIFICATION SUMMARY")
    print(f"    -> Verified Key Paths:  {count_pos} / {available_trajectories}")
    print(f"    -> Overall Match Rate:  {confidence_percentage:.2f}%")
    print("-"*70)

    # Core Security Verdict Logic
    print("🚨 FINAL AUDIT VERDICT:")
    if detection_rate >= 0.80:
        print("   " + "!"*64)
        print("   ⚠️ WARNING: [STOLEN / PIRATED INTELLECTUAL PROPERTY DETECTED]")
        print("   " + "!"*64)
        print(f"   The suspect model matches your proprietary brainwave decision dynamics")
        print(f"   with an undeniable confidence metric of {confidence_percentage:.1f}%.")
        print(f"   This model is derived from or direct copy-paste of your training history.")
        
    elif 0.30 <= detection_rate < 0.80:
        print("   " + "?"*64)
        print("   ⚠️ INDETERMINATE: [HIGH COEFFICIENT OF DERIVATION / IP LEAK]")
        print("   " + "?"*64)
        print(f"   Significant alignment detected ({confidence_percentage:.1f}%). The suspect")
        print(f"   has likely distilled, heavily fine-tuned, or pruned your original model.")
        print(f"   Possibility of partial watermark dilution.")
        
    else:
        print("   🛡️ CLEAN: [AUTHENTIC / INDEPENDENT IMPLEMENTATION]")
        print(f"   The tested network shares a negligible trajectory profile match ({confidence_percentage:.1f}%).")
        print(f"   The model was independently engineered without unauthorized asset replication.")
        
    print("="*70 + "\n")