#!/usr/bin/env python
# -*- coding: utf-8
"""
StarDist-based cell segmentation and foci detection.

For each TIFF in input_folder:
 1. Segment cells with StarDist2D.
 2. For each cell:
    - Crop the cell’s bounding box.
    - Remove smooth background via white top-hat.
    - Threshold with Otsu’s method.
    - Remove small objects.
    - Count the remaining connected components as foci.
 3. Save:
    - the full-label “cell mask” as a 16-bit TIFF,
    - an overlay PNG (cells semi‑transparent + foci centroids),
    - a CSV listing cell_id,foci_count.
"""

import os
import csv
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.morphology import white_tophat, disk, remove_small_objects
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops


# --- USER PARAMETERS ---
input_folder   = r"D:/DNA_damage/stardist_test2"
pattern        = "*.tif"
bg_disk_radius = 30      # for white top‑hat (should exceed foci diameter)
min_foci_area  = 2       # drop connected components smaller than this
min_cell_area  = 50      # ignore tiny debris
blur_sigma     = 2     # Gaussian blur σ (in pixels)
# -----------------------

# Load all TIFF files
files = sorted(glob(os.path.join(input_folder, pattern)))
if not files:
    raise RuntimeError(f"No files found in {input_folder}/{pattern}")

# Load pretrained StarDist2D model
model = StarDist2D.from_pretrained('2D_demo')

# Structuring element for top‑hat
selem = disk(bg_disk_radius)

for filepath in files:
    base = os.path.splitext(os.path.basename(filepath))[0]
    print(f"Processing {base}...")

    # 1. Read & normalize image
    img = imread(filepath)
    img_norm = normalize(img, 1, 99.8, axis=(0,1))

    # 2. Segment with StarDist
    labels, _ = model.predict_instances(img_norm)

    # 3. Prepare per-cell foci counting
    cell_ids    = []
    foci_counts = []

    for region in regionprops(labels, intensity_image=img_norm):
        if region.area < min_cell_area:
            continue

        cell_id = region.label
        cell_ids.append(cell_id)

        # a) Crop to bounding box
        minr, minc, maxr, maxc = region.bbox
        cell = img_norm[minr:maxr, minc:maxc]

        # 2) Mask out non‐nuclear pixels
        mask = region.image              # boolean array same shape as `cell`
        cell_masked = cell.copy()
        cell_masked[~mask] = 0

        # # add Gaussian blur **before** foci detection
        # cell = gaussian(cell,
        #                 sigma=blur_sigma,
        #                 preserve_range=True)

        # 2) Estimate background as the median intensity *within* the mask
        bg_val = np.median(cell[ mask ])   # mask = region.image boolean

        # 3) Subtract and zero‐clip
        cell_corr = cell - bg_val
        cell_corr[cell_corr < 0] = 0

        # # 5. Save the overlay image
        # fig, ax = plt.subplots(1,1, figsize=(6,6))
        # # show normalized image
        # ax.imshow(cell_corr if cell_corr.ndim==2 else cell_corr[...,0], cmap='gray')

        # c) Otsu thresholding
        if cell_corr.max() > 0:
            thr = threshold_otsu(cell_corr)
            foci_mask = cell_corr > thr
        else:
            foci_mask = np.zeros_like(cell_corr, dtype=bool)

        # d) Remove small objects
        foci_mask = remove_small_objects(foci_mask, min_size=min_foci_area)

        # e) Label + count foci
        foci_lbl   = label(foci_mask)
        foci_props = regionprops(foci_lbl)
        count = len(foci_props)
        foci_counts.append(count)

    # 4. Save the cell mask
    mask_path = os.path.join(input_folder, f"{base}_cell_mask.tif")
    imwrite(mask_path, labels.astype(np.uint16))
    print(f"  → Saved cell mask: {mask_path}")

    # 5. Save the overlay image
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    # show normalized image
    ax.imshow(img_norm if img_norm.ndim==2 else img_norm[...,0], cmap='gray')
    # overlay semi‑transparent cells
    ax.imshow(labels, cmap='jet', alpha=0.3)
    # overlay foci centroids
    for region in regionprops(labels, intensity_image=img_norm):
        if region.area < min_cell_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        cell = img_norm[minr:maxr, minc:maxc]
        mask = region.image              # boolean array same shape as `cell`
        cell_masked = cell.copy()
        cell_masked[~mask] = 0
        bg_val = np.median(cell[ mask ])   # mask = region.image boolean
        cell_corr = cell - bg_val
        cell_corr[cell_corr < 0] = 0
        if cell_corr.max() > 0:
            thr = threshold_otsu(cell_corr)
            foci_mask = cell_corr > thr
        else:
            foci_mask = np.zeros_like(cell_corr, dtype=bool)
        foci_mask = remove_small_objects(foci_mask, min_size=min_foci_area)
        for fp in regionprops(label(foci_mask)):
            y0, x0 = fp.centroid
            ax.plot(minc + x0, minr + y0, 'ro', markersize=4)
    ax.axis('off')
    overlay_path = os.path.join(input_folder, f"{base}_overlay.png")
    plt.tight_layout()
    fig.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved overlay: {overlay_path}")

    # 6. Save foci counts to CSV
    csv_path = os.path.join(input_folder, f"{base}_foci_counts.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_id','foci_count'])
        for cid, cnt in zip(cell_ids, foci_counts):
            writer.writerow([cid, cnt])
    print(f"  → Saved counts: {csv_path}")

print("Done.")
