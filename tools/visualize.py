# ------------------------------------------------------------------------------
# Visualization Script for PIDNet
# Loads a trained model and saves side-by-side comparisons:
# [Input Image] | [Ground Truth] | [Prediction]
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import _init_paths
import models
import datasets
from configs import config
from configs import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize PIDNet Results')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model-path', help='path to best.pt', required=True, type=str)
    parser.add_argument('--num-samples', help='number of images to visualize', default=10, type=int)
    parser.add_argument('--output-dir', help='where to save visualizations', default='vis_results', type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args

# Cityscapes color palette (for 19 classes, we will use this for our 15 too)
# This matches the official Cityscapes color scheme
CITYSCAPES_PALETTE = [
    128, 64, 128,  # 0: road
    244, 35, 232,  # 1: sidewalk
    70, 70, 70,    # 2: building
    102, 102, 156, # 3: wall
    190, 153, 153, # 4: fence
    153, 153, 153, # 5: pole
    250, 170, 30,  # 6: traffic light
    220, 220, 0,   # 7: traffic sign
    107, 142, 35,  # 8: vegetation
    152, 251, 152, # 9: terrain
    70, 130, 180,  # 10: sky
    220, 20, 60,   # 11: person
    255, 0, 0,     # 12: rider
    0, 0, 142,     # 13: car
    0, 0, 70,      # 14: truck
    0, 60, 100,    # 15: bus (ignore in your case)
    0, 80, 100,    # 16: train (ignore)
    0, 0, 230,     # 17: motorcycle (ignore)
    119, 11, 32    # 18: bicycle (ignore)
]

def colorize_mask(mask):
    """
    Converts a single-channel mask (H, W) to a color image (H, W, 3)
    using the Cityscapes palette.
    """
    # Create an empty color image
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # We iterate through your 15 classes
    for label_id in range(15): # 0 to 14
        # Find pixels with this label
        idx = (mask == label_id)
        # Paint them with the corresponding color from the palette
        color_mask[idx] = CITYSCAPES_PALETTE[label_id*3 : label_id*3 + 3]
        
    # Note: Ignore labels (255) will remain black (0,0,0)
    return color_mask

def main():
    args = parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 1. Build Model
    print(f"=> Creating model: {config.MODEL.NAME}")
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)
    
    # 2. Load Trained Weights
    print(f"=> Loading weights from: {args.model_path}")
    if os.path.isfile(args.model_path):
        pretrained_dict = torch.load(args.model_path, map_location='cpu')
        model_dict = model.state_dict()
        
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            
        # --- CRITICAL FIX START ---
        # 1. Remove 'module.' if it exists (DataParallel)
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        
        # 2. Remove 'model.' if it exists (FullModel wrapper)
        # This is the step that was missing!
        pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items()}
        # --- CRITICAL FIX END ---
        
        # Filter out keys that don't belong
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        
        # Update the model
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"=> Loaded {len(pretrained_dict)} keys successfully.")
    else:
        print(f"Error: No checkpoint found at {args.model_path}")
        return

    model = model.cuda()
    model.eval()

    # 3. Prepare Validation Data
    # We use the validation set to visualize results
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True, # Randomize to see different images
        num_workers=0, # Avoid multiprocessing issues in notebooks
        pin_memory=True)

    print(f"=> Visualizing {args.num_samples} samples...")
    
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            if index >= args.num_samples:
                break
                
            image, label, _, _, name = batch
            
            # Move image to GPU
            model_input = image.cuda()
            
            # Run Inference
            # PIDNet returns a list [aux_output, main_output] (or just main_output)
            # We want the main output (index 1 or just the output itself)
            output = model(model_input)
            
            if isinstance(output, (list, tuple)):
                # Output 1 is the main branch (the good one)
                pred = output[1] 
            else:
                pred = output

            # Interpolate to original size
            size = label.size()
            pred = F.interpolate(
                pred, size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            
            # Get class predictions (argmax)
            pred = torch.argmax(pred, dim=1).cpu().numpy()[0] # Shape: (H, W)
            label = label.numpy()[0] # Shape: (H, W)
            
            # --- Visualization Logic ---
            
            # 1. Original Image (Denormalize)
            # Image comes in as (1, 3, H, W) normalized
            # We need to reverse normalization: img = (img * std) + mean
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            img_vis = image[0].permute(1, 2, 0).numpy() # (H, W, 3)
            img_vis = (img_vis * std + mean) * 255.0
            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV saving, or keep RGB for matplotlib
            # Cityscapes images are usually BGR when read by cv2, but dataset converts to RGB
            # Let's keep it RGB for PIL
            
            # 2. Ground Truth Mask (Colorized)
            gt_vis = colorize_mask(label)
            
            # 3. Prediction Mask (Colorized)
            pred_vis = colorize_mask(pred)
            
            # 4. Combine Side-by-Side
            # Concatenate horizontally: [Image | GT | Pred]
            # Ensure sizes match exactly (they should)
            combined = np.concatenate((img_vis, gt_vis, pred_vis), axis=1)
            
            # 5. Save Result
            save_path = os.path.join(args.output_dir, f"{name[0]}_vis.png")
            Image.fromarray(combined).save(save_path)
            
    print(f"\n=> Done! Check the '{args.output_dir}' folder for your results.")

if __name__ == '__main__':
    main()