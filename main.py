# -*- coding: utf-8 -*-
import argparse
import sys
import os

# Adjust path so Python can seamlessly find our custom modules inside Colab
sys.path.append(os.path.abspath("./ADV-TRA-master"))

from utils.data_process import allocate_data
from utils.adv_gen import generate_trajectory, verify_trajectory

def main():
    parser = argparse.ArgumentParser(description="ADV-TRA Engine for Physics-Informed BCI Fingerprinting")
    
    # Core Pipeline Operational Mode Flag
    parser.add_argument('--mode', type=str, default='generate', choices=['allocate', 'generate', 'verify'],
                        help="Pipeline routine step: 'allocate' data, 'generate' fingerprints, or 'verify' a suspect model")
    
    # Dataset & Storage Target Paths
    parser.add_argument('--dataset', type=str, default='bci_sub2a', help="Dataset target folder layout identifier")
    parser.add_argument('--data_path', type=str, default='./ADV-TRA-master/data', help="Root directory location for loaded data")
    parser.add_argument('--model_path', type=str, default='./ADV-TRA-master/model_path', help="Directory storing your clean source model weights")
    parser.add_argument('--fingerprint_path', type=str, default='./ADV-TRA-master/fingerprint_path', help="Directory to save generated trajectories")
    parser.add_argument('--suspect_path', type=str, default='./ADV-TRA-master/model_path/bci_sub2a/suspect_model.pth', help="Path to suspect model weights during verification mode")
    
    # Allocation Size Bounds (Used in --mode allocate)
    parser.add_argument('--num_train', type=int, default=1000, help="Number of baseline training window samples to allocate")
    parser.add_argument('--num_attack', type=int, default=500, help="Number of anchor fingerprint base seed slices")
    parser.add_argument('--shuffle', type=bool, default=True, help="Shuffle dataset before allocation partitioning")
    
    # Model & Optimization Task Parameters (Used in --mode generate / verify)
    parser.add_argument('--num_classes', type=int, default=4, help="Number of target motor imagery classification options (Graz 2a = 4)")
    parser.add_argument('--tra_classes', type=int, default=4, help="Number of classes the adversarial trajectory path must traverse")
    parser.add_argument('--length', type=int, default=20, help="Total sequence step length of the bilateral trajectory (Must be an even integer)")
    parser.add_argument('--num_trajectories', type=int, default=10, help="Number of core independent fingerprint trajectories to generate")
    
    # Hyperparameters from Original Paper Repository
    parser.add_argument('--device', type=str, default='cuda', help="Computation hardware target ('cuda' or 'cpu')")
    parser.add_argument('--initial_stepsize', type=float, default=0.05, help="Starting step size for optimization boundary exploration")
    parser.add_argument('--max_iteration', type=int, default=300, help="Max optimization epochs to refine step sizes per trajectory loop")
    parser.add_argument('--tra_lr', type=float, default=0.001, help="Learning rate for step size optimization tuning updates")
    parser.add_argument('--factor_re', type=float, default=0.90, help="Decay factor for step sizes approaching decision boundaries")
    parser.add_argument('--factor_lc', type=float, default=1.05, help="Boundary crossing adjustment scaling multiplier parameter")
    parser.add_argument('--threshold', type=float, default=0.15, help="Mutation variance rate threshold tolerance for verifying ownership")

    args = parser.parse_args()
    
    # Convert device flag automatically if CUDA is missing or explicitly unavailable
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    # Route operation steps cleanly based on mode choice selection
    if args.mode == 'allocate':
        print("\n========== STARTING DATA ALLOCATION PIPELINE ==========")
        allocate_data(args)
        print("=======================================================")
        
    elif args.mode == 'generate':
        print("\n========== STARTING ADVERSARIAL TRAJECTORY GENERATION ==========")
        # Ensure the clean source model weights file path layout target exists before calling generator
        src_check_path = f"{args.model_path}/{args.dataset}/source_model.pth"
        if not os.path.exists(src_check_path):
            # Dynamic directory safety creation fallback
            os.makedirs(os.path.dirname(src_check_path), exist_ok=True)
            print(f"⚠️ Notice: Please ensure your trained '78.pth' weights file is saved or copied to: {src_check_path}")
            print("--> Pipeline paused. Save your clean source model weights file to the path above and re-run.")
            return
            
        generate_trajectory(args)
        print("================================================================")
        
    elif args.mode == 'verify':
        print("\n========== STARTING SUSPECT MODEL FINGERPRINT VERIFICATION ==========")
        if not os.path.exists(args.suspect_path):
            print(f"❌ Error: Could not find target suspect weights file at path location: {args.suspect_path}")
            return
        verify_trajectory(args)
        print("=====================================================================")

if __name__ == '__main__':
    main()
