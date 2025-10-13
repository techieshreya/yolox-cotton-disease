# utils/utils.py
import torch
import numpy as np
import os
import random
import shutil
from collections import defaultdict


def collate_fn(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Handles variable-sized bounding boxes.
    """
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


def box_iou(boxes1, boxes2):
    """
    Calculates the Intersection over Union (IoU) of two sets of bounding boxes.
    """
    # Implementation of box_iou (as provided earlier)
    # ... (Code for box_iou)
    pass # Replace with your box_iou implementation


def stratified_split_voc(input_dir, output_train_dir, output_val_dir, val_ratio=0.12, seed=42):
	"""
	Create a stratified split of VOC-style dataset (images + XML) by class labels.
	- input_dir: directory with mixed images and xml annotations
	- output_train_dir/output_val_dir: target directories (created if missing)
	- val_ratio: fraction of images per class to put in validation
	
	Stratification is by image-level labels (any class appearing in the image).
	"""
	random.seed(seed)
	os.makedirs(output_train_dir, exist_ok=True)
	os.makedirs(output_val_dir, exist_ok=True)

	# Collect images and associated classes
	image_to_classes = {}
	classes = set()
	for f in os.listdir(input_dir):
		if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
			continue
		name = os.path.splitext(f)[0]
		xml = os.path.join(input_dir, name + '.xml')
		if not os.path.exists(xml):
			continue
		# naive parse
		with open(xml, 'r', encoding='utf-8', errors='ignore') as xf:
			content = xf.read().lower()
			img_classes = []
			for c in ['curl_stage1', 'curl_stage2', 'healthy', 'leaf_enation', 'sooty']:
				if f'<name>{c}</name>' in content:
					img_classes.append(c)
					classes.add(c)
		image_to_classes[f] = img_classes if img_classes else ['healthy']

	# Group by a canonical single class for stratification: choose the rarest present label per image
	class_counts = defaultdict(int)
	for _, lbls in image_to_classes.items():
		for l in set(lbls):
			class_counts[l] += 1

	def rarest_label(lbls):
		return sorted(lbls, key=lambda x: class_counts[x])[0]

	buckets = defaultdict(list)
	for img, lbls in image_to_classes.items():
		buckets[rarest_label(lbls)].append(img)

	train_set, val_set = set(), set()
	for c, imgs in buckets.items():
		random.shuffle(imgs)
		n_val = max(1, int(len(imgs) * val_ratio))
		val_imgs = set(imgs[:n_val])
		train_imgs = set(imgs[n_val:])
		val_set |= val_imgs
		train_set |= train_imgs

	# Move files
	for img in train_set:
		name = os.path.splitext(img)[0]
		shutil.copy2(os.path.join(input_dir, img), os.path.join(output_train_dir, img))
		xml = name + '.xml'
		shutil.copy2(os.path.join(input_dir, xml), os.path.join(output_train_dir, xml))
	for img in val_set:
		name = os.path.splitext(img)[0]
		shutil.copy2(os.path.join(input_dir, img), os.path.join(output_val_dir, img))
		xml = name + '.xml'
		shutil.copy2(os.path.join(input_dir, xml), os.path.join(output_val_dir, xml))

	return {
		'train_count': len(train_set),
		'val_count': len(val_set),
		'by_class': {c: len(buckets.get(c, [])) for c in sorted(classes)}
	}