# datasets/pascal_voc.py
import os
import cv2
import numpy as np
import torch
from .base_dataset import BaseDataset

class PascalVOC(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=16,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=512, 
                 crop_size=(512, 512), 
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(PascalVOC, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        
        self.class_weights = None

        self.bd_dilate_size = bd_dilate_size

        # Read the list file
        if not os.path.exists(list_path):
             possible_path = os.path.join(root, list_path)
             if os.path.exists(possible_path):
                 list_path = possible_path
             else:
                 raise FileNotFoundError(f"Could not find train/val list at {list_path}")

        self.img_list = [line.strip().split() for line in open(list_path)]
        self.files = self.read_files()

    def read_files(self):
        files = []
        for item in self.img_list:
            raw_path = item[0]
            # --- FIX: Robustly extract just the ID ---
            # This strips folder paths and extensions so we get just the ID
            name = os.path.splitext(os.path.basename(raw_path))[0]
            files.append({"name": name})
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        image_path = os.path.join(self.root, 'JPEGImages', name + '.jpg')
        label_path = os.path.join(self.root, 'SegmentationClass', name + '.png')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if label is None:
            # Fallback: Try searching for the label if exact path fails (robustness)
            raise FileNotFoundError(f"Label not found: {label_path}")

        # --- VOC 15-1 Masking ---
        label[label > 15] = self.ignore_label
        # ------------------------

        size = image.shape
        
        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, 
                                edge_size=self.bd_dilate_size)

        # --- Padding Logic ---
        h, w = image.shape[1], image.shape[2]
        
        if image.ndim == 3 and image.shape[0] != 3: 
             h, w = image.shape[0], image.shape[1]

        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            if image.ndim == 3 and image.shape[0] == 3:
                 image = np.pad(image, ((0,0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                 label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=self.ignore_label)
                 edge  = np.pad(edge,  ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            else: 
                 image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                 label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
                 edge  = cv2.copyMakeBorder(edge, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred